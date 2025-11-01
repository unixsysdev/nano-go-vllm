package main

import (
    "flag"
    "fmt"
    "log"
    "os"
    "sort"

    "github.com/unixsysdev/nano-go-vllm/internal/config"
    "github.com/unixsysdev/nano-go-vllm/internal/engine"
    "github.com/unixsysdev/nano-go-vllm/internal/models"
    "github.com/unixsysdev/nano-go-vllm/internal/sampling"
    "github.com/unixsysdev/nano-go-vllm/pkg/tokenizer"
    "github.com/unixsysdev/nano-go-vllm/internal/tensor"
)

func main() {
    fs := flag.NewFlagSet("nanovllm", flag.ExitOnError)
    maxTokens := fs.Int("max-tokens", 64, "maximum new tokens to generate")
    temperature := fs.Float64("temperature", 0.7, "sampling temperature")
    topP := fs.Float64("top-p", 0.95, "nucleus sampling probability mass (0-1)")
    topK := fs.Int("top-k", 50, "top-k sampling (0 = disabled)")
    repPenalty := fs.Float64("repetition-penalty", 1.1, "repetition penalty (>1 to penalize repeats)")
    presencePenalty := fs.Float64("presence-penalty", 0.0, "presence penalty (penalize seen tokens)")
    frequencyPenalty := fs.Float64("frequency-penalty", 0.0, "frequency penalty (per occurrence)")
    stream := fs.Bool("stream", false, "stream tokens as they are generated")
    verify := fs.Bool("verify", false, "print top logits for the last token (no sampling)")
    _ = fs.Parse(os.Args[1:])

    args := fs.Args()
    if len(args) < 1 {
        fmt.Println("Usage: nanovllm [flags] <model_path> [prompt]")
        fs.PrintDefaults()
        os.Exit(1)
    }

    modelPath := args[0]
    prompt := "Hello, how are you?"
    if len(args) > 1 {
        prompt = args[1]
    }

    // Initialize engine
    llmEngine, err := engine.NewLLMEngine(
        modelPath,
        config.WithEnforceEager(true),
        config.WithTensorParallelSize(1),
    )
    if err != nil { log.Fatalf("Failed to initialize LLM engine: %v", err) }

	// Set sampling parameters
    params := &sampling.SamplingParams{
        Temperature: float32(*temperature),
        MaxTokens:   *maxTokens,
        IgnoreEOS:   false,
        TopP:        float32(*topP),
        TopK:        *topK,
        RepetitionPenalty: float32(*repPenalty),
        PresencePenalty:   float32(*presencePenalty),
        FrequencyPenalty:  float32(*frequencyPenalty),
    }

    tok, _ := tokenizer.NewTokenizer(modelPath)
    if *verify {
        // Build model directly and print last-token top logits
        cfg, err := config.LoadConfig(modelPath)
        if err != nil { log.Fatalf("config: %v", err) }
        mdl, err := models.NewQwenModel(cfg)
        if err != nil { log.Fatalf("model: %v", err) }
        ids, err := tok.Encode(prompt)
        if err != nil { log.Fatalf("encode: %v", err) }
        // Prepare input tensors [T] and positions [T]
        T := len(ids)
        idT, _ := tensor.NewTensor([]int{T}, tensor.Int64, tensor.CPU)
        posT, _ := tensor.NewTensor([]int{T}, tensor.Int64, tensor.CPU)
        idBuf := idT.Data().Data().([]int64)
        posBuf := posT.Data().Data().([]int64)
        for i, v := range ids { idBuf[i] = int64(v); posBuf[i] = int64(i) }
        logits, err := mdl.Forward(idT, posT)
        if err != nil { log.Fatalf("forward: %v", err) }
        shape := logits.Shape()
        if len(shape) != 2 { log.Fatalf("unexpected logits shape: %v", shape) }
        V := shape[1]
        data := logits.Data().Data().([]float32)
        last := data[(T-1)*V : T*V]
        // find top-10
        type kv struct{ id int; logit float32 }
        top := make([]kv, V)
        for i := 0; i < V; i++ { top[i] = kv{i, last[i]} }
        sort.Slice(top, func(i,j int) bool { return top[i].logit > top[j].logit })
        if len(top) > 10 { top = top[:10] }
        fmt.Printf("Prompt: %s\n", prompt)
        fmt.Println("Top logits (id, logit, token):")
        for _, k := range top {
            s, _ := tok.Decode([]int{k.id})
            fmt.Printf("%d\t%.4f\t%s\n", k.id, k.logit, s)
        }
        return
    }
    if *stream {
        // streaming: add request then step
        if err := llmEngine.AddRequest(prompt, params); err != nil { log.Fatalf("add request: %v", err) }
        fmt.Printf("Prompt: %s\n", prompt)
        fmt.Print("Output: ")
        var outTokens []int
        for !llmEngine.IsFinished() {
            stepOut, err := llmEngine.Step()
            if err != nil { log.Fatalf("step failed: %v", err) }
            for _, so := range stepOut {
                // print only new tokens (so.TokenIDs is completion set)
                if len(so.TokenIDs) > len(outTokens) {
                    new := so.TokenIDs[len(outTokens):]
                    outTokens = append(outTokens, new...)
                    if s, err := tok.Decode(new); err == nil { fmt.Print(s) }
                }
            }
        }
        fmt.Printf("\nToken IDs: %v\n", outTokens)
    } else {
        // Generate text (non-streaming)
        outputs, err := llmEngine.Generate([]string{prompt}, []*sampling.SamplingParams{params})
        if err != nil { log.Fatalf("Generation failed: %v", err) }
        fmt.Printf("Prompt: %s\n", prompt)
        fmt.Printf("Output: %s\n", outputs[0].Text)
        fmt.Printf("Token IDs: %v\n", outputs[0].TokenIDs)
    }
}
