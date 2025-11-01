package tokenizer

import (
    "bytes"
    "encoding/json"
    "fmt"
    "os"
    "os/exec"
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

    added       map[string]int
    addedSorted []string

    pattern *regexp.Regexp
    bpeCache map[string][]string
}

// NewTokenizer loads a HF-like tokenizer.json with model.type == "BPE" and optional ByteLevel pre_tokenizer
func NewTokenizer(modelPath string) (Tokenizer, error) {
    // Prefer external HF tokenizer if available for immediate parity
    if ext, err := newExternal(modelPath); err == nil {
        return ext, nil
    }
    tokPath := filepath.Join(modelPath, "tokenizer.json")
    data, err := os.ReadFile(tokPath)
    if err != nil {
        return nil, fmt.Errorf("read tokenizer.json: %w", err)
    }
    var cfg struct {
        Model struct {
            Type  string            `json:"type"`
            Vocab map[string]int    `json:"vocab"`
            MergesRaw []interface{} `json:"merges"`
            UnkToken string         `json:"unk_token"`
        } `json:"model"`
        PreTokenizer struct {
            Type string `json:"type"`
            AddPrefixSpace bool `json:"add_prefix_space"`
        } `json:"pre_tokenizer"`
        AddedTokens []struct {
            ID int `json:"id"`
            Content string `json:"content"`
            Special bool `json:"special"`
        } `json:"added_tokens"`
    }
    if err := json.Unmarshal(data, &cfg); err != nil {
        return nil, fmt.Errorf("parse tokenizer.json: %w", err)
    }
    if strings.ToUpper(cfg.Model.Type) != "BPE" {
        return nil, fmt.Errorf("unsupported tokenizer model type: %s", cfg.Model.Type)
    }

    // Build ranks
    var merges []string
    for _, it := range cfg.Model.MergesRaw {
        switch v := it.(type) {
        case string:
            merges = append(merges, v)
        case []interface{}:
            if len(v) == 2 {
                a, _ := v[0].(string)
                b, _ := v[1].(string)
                if a != "" && b != "" { merges = append(merges, a+" "+b) }
            }
        }
    }
    ranks := make(map[[2]string]int, len(merges))
    for i, m := range merges {
        parts := strings.Split(m, " ")
        if len(parts) != 2 { continue }
        ranks[[2]string{parts[0], parts[1]}] = i
    }

    // Build idToToken slice including added tokens
    vocab := make(map[string]int, len(cfg.Model.Vocab)+len(cfg.AddedTokens))
    for k, v := range cfg.Model.Vocab { vocab[k] = v }
    for _, a := range cfg.AddedTokens { vocab[a.Content] = a.ID }
    maxID := -1
    for _, id := range vocab { if id > maxID { maxID = id } }
    idToTok := make([]string, maxID+1)
    for tok, id := range vocab { if id >= 0 && id < len(idToTok) { idToTok[id] = tok } }

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

    // Added tokens sorted by length desc for greedy longest match
    added := make(map[string]int)
    for _, a := range cfg.AddedTokens { added[a.Content] = a.ID }
    var addedSorted []string
    for s := range added { addedSorted = append(addedSorted, s) }
    sort.Slice(addedSorted, func(i,j int) bool { return len(addedSorted[i]) > len(addedSorted[j]) })

    bt, bd := bytesToUnicode()
    // GPT-2 ByteLevel pretokenizer regex
    pat := regexp.MustCompile(`(?i)'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`)
    return &bpeTokenizer{
        vocab: vocab,
        idToToken: idToTok,
        mergesRank: ranks,
        addPrefixSpace: cfg.PreTokenizer.Type == "ByteLevel" && cfg.PreTokenizer.AddPrefixSpace,
        eosID: eosID,
        unkID: unkID,
        byteEncoder: bt,
        byteDecoder: bd,
        added: added,
        addedSorted: addedSorted,
        pattern: pat,
        bpeCache: make(map[string][]string),
    }, nil
}

// Encode runs a simplified ByteLevel pretokenization + BPE
func (t *bpeTokenizer) Encode(text string) ([]int, error) {
    var ids []int
    if t.addPrefixSpace && (len(text) > 0 && text[0] != ' ') {
        text = " " + text
    }
    pos := 0
    for pos < len(text) {
        // Greedy special tokens
        matched := false
        for _, tok := range t.addedSorted {
            if strings.HasPrefix(text[pos:], tok) {
                ids = append(ids, t.added[tok])
                pos += len(tok)
                matched = true
                break
            }
        }
        if matched { continue }
        // Bound by next special occurrence
        next := len(text)
        for _, tok := range t.addedSorted {
            if i := strings.Index(text[pos:], tok); i >= 0 {
                if pos+i < next { next = pos+i }
            }
        }
        segment := text[pos:next]
        toks := t.pattern.FindAllString(segment, -1)
        for _, tk := range toks {
            if tk == "" { continue }
            ids = append(ids, t.encodeWord(tk)...)
        }
        pos = next
    }
    return ids, nil
}

func (t *bpeTokenizer) encodeWord(word string) []int {
    // ByteLevel bytes->unicode mapping
    raw := []byte(word)
    var sb strings.Builder
    sb.Grow(len(raw))
    for _, b := range raw { sb.WriteRune(t.byteEncoder[b]) }
    token := sb.String()
    if v, ok := t.bpeCache[token]; ok {
        return t.tokensToIDs(v)
    }
    pieces := t.applyBPE(token)
    t.bpeCache[token] = pieces
    return t.tokensToIDs(pieces)
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

// --- BPE internals ---
func (t *bpeTokenizer) applyBPE(token string) []string {
    if token == "" { return nil }
    // split into runes (as strings)
    symbols := make([]string, 0, len(token))
    for _, r := range token { symbols = append(symbols, string(r)) }
    if len(symbols) == 1 { return symbols }
    pairs := getPairs(symbols)
    for {
        if len(pairs) == 0 { break }
        var best [2]string
        bestRank := int(^uint(0) >> 1)
        found := false
        for p := range pairs {
            if r, ok := t.mergesRank[p]; ok {
                if r < bestRank { bestRank = r; best = p; found = true }
            }
        }
        if !found { break }
        // merge occurrences
        merged := make([]string, 0, len(symbols))
        i := 0
        for i < len(symbols) {
            j := indexPair(symbols, i, best)
            if j == -1 {
                merged = append(merged, symbols[i:]...)
                break
            }
            merged = append(merged, symbols[i:j]...)
            merged = append(merged, symbols[j]+symbols[j+1])
            i = j + 2
        }
        symbols = merged
        if len(symbols) == 1 { break }
        pairs = getPairs(symbols)
    }
    return symbols
}

func getPairs(symbols []string) map[[2]string]struct{} {
    pairs := make(map[[2]string]struct{})
    for i := 0; i < len(symbols)-1; i++ {
        pairs[[2]string{symbols[i], symbols[i+1]}] = struct{}{}
    }
    return pairs
}

func indexPair(symbols []string, start int, pair [2]string) int {
    for i := start; i < len(symbols)-1; i++ {
        if symbols[i] == pair[0] && symbols[i+1] == pair[1] { return i }
    }
    return -1
}

func (t *bpeTokenizer) tokensToIDs(tokens []string) []int {
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

// External tokenizer uses Python tokenizers library via scripts/tokenizer_adapter.py
type external struct { modelPath string; py string; script string; eos int }

func pickPython() (string, error) {
    // Try venv python first
    cand := []string{".venv/bin/python", ".venv/bin/python3", "python3", "python"}
    for _, p := range cand {
        if _, err := os.Stat(p); err == nil { return p, nil }
        if exe, err := exec.LookPath(p); err == nil { return exe, nil }
    }
    return "", fmt.Errorf("python not found")
}

func newExternal(modelPath string) (*external, error) {
    py, err := pickPython()
    if err != nil { return nil, err }
    script := filepath.Join("scripts", "tokenizer_adapter.py")
    if _, err := os.Stat(script); err != nil { return nil, err }
    // Try to read eos from config.json
    eos := -1
    if data, err := os.ReadFile(filepath.Join(modelPath, "config.json")); err == nil {
        var cfg map[string]interface{}
        if json.Unmarshal(data, &cfg) == nil {
            if v, ok := cfg["eos_token_id"].(float64); ok { eos = int(v) }
        }
    }
    return &external{modelPath: modelPath, py: py, script: script, eos: eos}, nil
}

func (e *external) Encode(text string) ([]int, error) {
    cmd := exec.Command(e.py, e.script, "encode", e.modelPath, text)
    var out bytes.Buffer
    cmd.Stdout = &out
    cmd.Stderr = &out
    if err := cmd.Run(); err != nil { return nil, fmt.Errorf("encode failed: %v: %s", err, out.String()) }
    var resp struct{ Ids []int `json:"ids"`; Error string `json:"error"` }
    if err := json.Unmarshal(out.Bytes(), &resp); err != nil { return nil, err }
    if resp.Error != "" { return nil, fmt.Errorf(resp.Error) }
    return resp.Ids, nil
}

func (e *external) Decode(tokenIDs []int) (string, error) {
    b, _ := json.Marshal(tokenIDs)
    cmd := exec.Command(e.py, e.script, "decode", e.modelPath, string(b))
    var out bytes.Buffer
    cmd.Stdout = &out
    cmd.Stderr = &out
    if err := cmd.Run(); err != nil { return "", fmt.Errorf("decode failed: %v: %s", err, out.String()) }
    var resp struct{ Text string `json:"text"`; Error string `json:"error"` }
    if err := json.Unmarshal(out.Bytes(), &resp); err != nil { return "", err }
    if resp.Error != "" { return "", fmt.Errorf(resp.Error) }
    return resp.Text, nil
}

func (e *external) GetEOS() int { return e.eos }

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
