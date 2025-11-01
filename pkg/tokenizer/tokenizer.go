package tokenizer

import (
    "encoding/json"
    "fmt"
    "os"
    "path/filepath"
    "regexp"
    "sort"
    "strings"
    "unicode/utf8"
)

// Tokenizer represents a tokenizer
type Tokenizer interface {
    Encode(text string) ([]int, error)
    Decode(tokenIDs []int) (string, error)
    GetEOS() int
}

// bpeTokenizer implements a minimal HF-compatible ByteLevel BPE
type bpeTokenizer struct {
    vocab       map[string]int
    idToToken  []string
    mergesRank map[[2]string]int
    addPrefixSpace bool
    eosID      int
    unkID      int

    byteEncoder map[byte]rune
    byteDecoder map[rune]byte
}

// NewTokenizer loads a HF-like tokenizer.json with model.type == "BPE" and optional ByteLevel pre_tokenizer
func NewTokenizer(modelPath string) (Tokenizer, error) {
    tokPath := filepath.Join(modelPath, "tokenizer.json")
    data, err := os.ReadFile(tokPath)
    if err != nil {
        return nil, fmt.Errorf("read tokenizer.json: %w", err)
    }
    var cfg struct {
        Model struct {
            Type  string            `json:"type"`
            Vocab map[string]int    `json:"vocab"`
            Merges []string         `json:"merges"`
            UnkToken string         `json:"unk_token"`
        } `json:"model"`
        PreTokenizer struct {
            Type string `json:"type"`
            AddPrefixSpace bool `json:"add_prefix_space"`
        } `json:"pre_tokenizer"`
    }
    if err := json.Unmarshal(data, &cfg); err != nil {
        return nil, fmt.Errorf("parse tokenizer.json: %w", err)
    }
    if strings.ToUpper(cfg.Model.Type) != "BPE" {
        return nil, fmt.Errorf("unsupported tokenizer model type: %s", cfg.Model.Type)
    }

    // Build ranks
    ranks := make(map[[2]string]int, len(cfg.Model.Merges))
    for i, m := range cfg.Model.Merges {
        parts := strings.Split(m, " ")
        if len(parts) != 2 { continue }
        ranks[[2]string{parts[0], parts[1]}] = i
    }

    // Build idToToken slice for decode
    maxID := -1
    for _, id := range cfg.Model.Vocab { if id > maxID { maxID = id } }
    idToTok := make([]string, maxID+1)
    for tok, id := range cfg.Model.Vocab { if id >= 0 && id < len(idToTok) { idToTok[id] = tok } }

    // Unk & EOS
    unkID := -1
    if cfg.Model.UnkToken != "" {
        if id, ok := cfg.Model.Vocab[cfg.Model.UnkToken]; ok { unkID = id }
    } else if id, ok := cfg.Model.Vocab["<unk>"]; ok { unkID = id }

    eosID := -1
    // Try config.json for eos_token_id
    if b, err := os.ReadFile(filepath.Join(modelPath, "config.json")); err == nil {
        var mc map[string]interface{}
        if json.Unmarshal(b, &mc) == nil {
            if v, ok := mc["eos_token_id"].(float64); ok { eosID = int(v) }
        }
    }

    bt, bd := bytesToUnicode()
    return &bpeTokenizer{
        vocab: cfg.Model.Vocab,
        idToToken: idToTok,
        mergesRank: ranks,
        addPrefixSpace: cfg.PreTokenizer.Type == "ByteLevel" && cfg.PreTokenizer.AddPrefixSpace,
        eosID: eosID,
        unkID: unkID,
        byteEncoder: bt,
        byteDecoder: bd,
    }, nil
}

// Encode runs a simplified ByteLevel pretokenization + BPE
func (t *bpeTokenizer) Encode(text string) ([]int, error) {
    // Basic whitespace split; add prefix space to each token except first if configured
    // In ByteLevel, spaces are significant; we approximate by splitting on spaces but keeping them by prefix space.
    // For better fidelity, a regex is often used; we'll split on runs of whitespace.
    re := regexp.MustCompile(`\S+|\s+`)
    parts := re.FindAllString(text, -1)
    var ids []int
    for i, p := range parts {
        if strings.TrimSpace(p) == "" {
            // Whitespace chunk - merge with next word as prefix space if applicable
            continue
        }
        token := p
        if t.addPrefixSpace && i > 0 {
            token = " " + token
        }
        for _, id := range t.encodeWord(token) {
            ids = append(ids, id)
        }
    }
    return ids, nil
}

func (t *bpeTokenizer) encodeWord(word string) []int {
    // Map to byte-level unicode
    var bytes []byte
    bytes = []byte(word)
    // map bytes -> unicode runes
    chars := make([]string, 0, len(bytes))
    for _, b := range bytes {
        r := t.byteEncoder[b]
        chars = append(chars, string(r))
    }

    // BPE merge
    tokens := bpeMerge(chars, t.mergesRank)
    // Map to ids
    out := make([]int, 0, len(tokens))
    for _, tok := range tokens {
        if id, ok := t.vocab[tok]; ok {
            out = append(out, id)
        } else if t.unkID >= 0 {
            out = append(out, t.unkID)
        }
    }
    return out
}

// Decode maps token ids back to text (approximate for ByteLevel)
func (t *bpeTokenizer) Decode(tokenIDs []int) (string, error) {
    // Proper ByteLevel decode: map rune->byte where possible
    buf := make([]byte, 0, len(tokenIDs)*4)
    for _, id := range tokenIDs {
        if id < 0 || id >= len(t.idToToken) { continue }
        tok := t.idToToken[id]
        for _, r := range tok {
            if b, ok := t.byteDecoder[r]; ok {
                buf = append(buf, b)
            } else {
                var tmp [4]byte
                n := utf8.EncodeRune(tmp[:], r)
                buf = append(buf, tmp[:n]...)
            }
        }
    }
    return string(buf), nil
}

func (t *bpeTokenizer) GetEOS() int { return t.eosID }

// bpeMerge performs BPE merges on slice of symbols using rank map
func bpeMerge(sym []string, ranks map[[2]string]int) []string {
    if len(sym) <= 1 { return sym }
    // Build initial pairs
    type pair struct { a, b string }
    for {
        // Find all pairs with ranks
        bestRank := int(^uint(0) >> 1) // max int
        bestIdx := -1
        for i := 0; i < len(sym)-1; i++ {
            r, ok := ranks[[2]string{sym[i], sym[i+1]}]
            if ok && r < bestRank {
                bestRank = r
                bestIdx = i
            }
        }
        if bestIdx == -1 {
            break
        }
        // Merge best pair
        merged := sym[bestIdx] + sym[bestIdx+1]
        news := make([]string, 0, len(sym)-1)
        news = append(news, sym[:bestIdx]...)
        news = append(news, merged)
        news = append(news, sym[bestIdx+2:]...)
        sym = news
        if len(sym) == 1 { break }
    }
    return sym
}

// bytesToUnicode constructs byte<->unicode mapping as in GPT-2 byte-level BPE
func bytesToUnicode() (map[byte]rune, map[rune]byte) {
    bs := []int{}
    for i := 33; i <= 126; i++ { bs = append(bs, i) }
    for i := 161; i <= 172; i++ { bs = append(bs, i) }
    for i := 174; i <= 255; i++ { bs = append(bs, i) }
    cs := make([]int, len(bs))
    copy(cs, bs)
    n := 0
    for b := 0; b < 256; b++ {
        if !contains(bs, b) {
            bs = append(bs, b)
            cs = append(cs, 256+n)
            n++
        }
    }
    sort.Ints(bs)
    // map in order of bs
    be := make(map[byte]rune)
    bd := make(map[rune]byte)
    for i, b := range bs {
        r := rune(cs[i])
        be[byte(b)] = r
        bd[r] = byte(b)
    }
    return be, bd
}

func contains(a []int, x int) bool {
    for _, v := range a { if v == x { return true } }
    return false
}
