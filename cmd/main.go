package main

import (
    "flag"
    "fmt"
    "log"
    "os"

    "github.com/unixsysdev/nano-go-vllm/internal/config"
    "github.com/unixsysdev/nano-go-vllm/internal/engine"
    "github.com/unixsysdev/nano-go-vllm/internal/sampling"
)

func main() {
    fs := flag.NewFlagSet("nanovllm", flag.ExitOnError)
    maxTokens := fs.Int("max-tokens", 64, "maximum new tokens to generate")
    temperature := fs.Float64("temperature", 0.7, "sampling temperature")
    topP := fs.Float64("top-p", 0.95, "nucleus sampling probability mass (0-1)")
    topK := fs.Int("top-k", 50, "top-k sampling (0 = disabled)")
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
    }

	// Generate text
	outputs, err := llmEngine.Generate([]string{prompt}, []*sampling.SamplingParams{params})
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	// Print output
	fmt.Printf("Prompt: %s\n", prompt)
    fmt.Printf("Output: %s\n", outputs[0].Text)
    fmt.Printf("Token IDs: %v\n", outputs[0].TokenIDs)
}
