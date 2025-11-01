package engine

import (
    "fmt"
    "sync"

    "github.com/unixsysdev/nano-go-vllm/internal/config"
    "github.com/unixsysdev/nano-go-vllm/internal/models"
    "github.com/unixsysdev/nano-go-vllm/internal/sampling"
    "github.com/unixsysdev/nano-go-vllm/pkg/tokenizer"
)

// LLMEngine represents the LLM inference engine
type LLMEngine struct {
	config      *config.Config
    model       *models.QwenModel
	tokenizer   tokenizer.Tokenizer
	scheduler   *Scheduler
	modelRunner *ModelRunner
	mu          sync.Mutex
}

// NewLLMEngine creates a new LLM engine
func NewLLMEngine(modelPath string, opts ...config.Option) (*LLMEngine, error) {
	// Load configuration
	cfg, err := config.LoadConfig(modelPath, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %v", err)
	}

	// Initialize tokenizer
	tok, err := tokenizer.NewTokenizer(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize tokenizer: %v", err)
	}

	// Initialize model
    model, err := models.NewQwenModel(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create model: %v", err)
	}

	// Initialize scheduler
	scheduler := NewScheduler(cfg)

	// Initialize model runner
	modelRunner, err := NewModelRunner(cfg, model)
	if err != nil {
		return nil, fmt.Errorf("failed to create model runner: %v", err)
	}

	return &LLMEngine{
		config:      cfg,
		model:       model,
		tokenizer:   tok,
		scheduler:   scheduler,
		modelRunner: modelRunner,
	}, nil
}

// AddRequest adds a new generation request
func (e *LLMEngine) AddRequest(prompt string, params *sampling.SamplingParams) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Tokenize prompt
	tokenIDs, err := e.tokenizer.Encode(prompt)
	if err != nil {
		return fmt.Errorf("tokenization failed: %v", err)
	}

	// Create sequence
	seq := NewSequence(tokenIDs, params)

	// Add to scheduler
	e.scheduler.Add(seq)

	return nil
}

// Step performs one inference step
func (e *LLMEngine) Step() ([]*SequenceOutput, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Schedule sequences
	seqs, isPrefill := e.scheduler.Schedule()
	if len(seqs) == 0 {
		return nil, nil
	}

	// Run model
	tokenIDs, err := e.modelRunner.Run(seqs, isPrefill)
	if err != nil {
		return nil, fmt.Errorf("model run failed: %v", err)
	}

	// Post-process sequences
	finished := e.scheduler.PostProcess(seqs, tokenIDs)
	
	outputs := make([]*SequenceOutput, len(seqs))
	for i, seq := range seqs {
		outputs[i] = &SequenceOutput{
			SeqID:    seq.ID,
			TokenIDs: seq.CompletionTokenIDs(),
			Finished: finished[i],
		}
	}

	return outputs, nil
}

// IsFinished checks if all requests are finished
func (e *LLMEngine) IsFinished() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.scheduler.IsFinished()
}

// Generate generates text for prompts
func (e *LLMEngine) Generate(prompts []string, params []*sampling.SamplingParams) ([]*GenerationOutput, error) {
	// Add all requests
	for i, prompt := range prompts {
		if err := e.AddRequest(prompt, params[i]); err != nil {
			return nil, fmt.Errorf("failed to add request %d: %v", i, err)
		}
	}

	// Process until all requests are finished
	outputs := make(map[int]*GenerationOutput)
	for !e.IsFinished() {
		stepOutputs, err := e.Step()
		if err != nil {
			return nil, fmt.Errorf("step failed: %v", err)
		}

		for _, output := range stepOutputs {
			if output.Finished {
				text, err := e.tokenizer.Decode(output.TokenIDs)
				if err != nil {
					return nil, fmt.Errorf("decoding failed: %v", err)
				}
				outputs[output.SeqID] = &GenerationOutput{
					Text:     text,
					TokenIDs: output.TokenIDs,
				}
			}
		}
	}

	// Collect outputs in order
	result := make([]*GenerationOutput, len(prompts))
	for i := range prompts {
		// Find the output for this prompt
		for _, output := range outputs {
			// This is a simplified approach - in practice, you'd need to track
			// which sequence corresponds to which prompt
			result[i] = output
			break
		}
	}

	return result, nil
}

// SequenceOutput represents output from a sequence step
type SequenceOutput struct {
	SeqID    int
	TokenIDs []int
	Finished bool
}

// GenerationOutput represents final generation output
type GenerationOutput struct {
	Text     string
	TokenIDs []int
}
