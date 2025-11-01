package main

import (
    "flag"
    "fmt"
    "log"
    "os"

    "github.com/unixsysdev/nano-go-vllm/internal/config"
    "github.com/unixsysdev/nano-go-vllm/internal/engine"
    "github.com/unixsysdev/nano-go-vllm/internal/sampling"
    "github.com/unixsysdev/nano-go-vllm/pkg/tokenizer"
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
	if err != nil {
		log.Fatalf("Failed to initialize LLM engine: %v", err)
	}

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
