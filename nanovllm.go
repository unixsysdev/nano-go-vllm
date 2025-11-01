package nanovllm

import (
    "github.com/unixsysdev/nano-go-vllm/internal/config"
    "github.com/unixsysdev/nano-go-vllm/internal/engine"
    "github.com/unixsysdev/nano-go-vllm/internal/sampling"
)

// LLM represents the main LLM interface
type LLM struct {
	engine *engine.LLMEngine
}

// NewLLM creates a new LLM instance
func NewLLM(modelPath string, opts ...config.Option) (*LLM, error) {
	engine, err := engine.NewLLMEngine(modelPath, opts...)
	if err != nil {
		return nil, err
	}
	
	return &LLM{
		engine: engine,
	}, nil
}

// Generate generates text for the given prompts
func (llm *LLM) Generate(prompts []string, params []*sampling.SamplingParams) ([]*GenerationOutput, error) {
	outputs, err := llm.engine.Generate(prompts, params)
	if err != nil {
		return nil, err
	}
	
	result := make([]*GenerationOutput, len(outputs))
	for i, output := range outputs {
		result[i] = &GenerationOutput{
			Text:     output.Text,
			TokenIDs: output.TokenIDs,
		}
	}
	
	return result, nil
}

// GenerationOutput represents the output from generation
type GenerationOutput struct {
	Text     string
	TokenIDs []int
}
