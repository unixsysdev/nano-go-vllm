package engine

import (
    "sync/atomic"

    "github.com/unixsysdev/nano-go-vllm/internal/sampling"
)

// SequenceStatus represents the status of a sequence
type SequenceStatus int

const (
	SequenceStatusWaiting SequenceStatus = iota
	SequenceStatusRunning
	SequenceStatusFinished
)

// Sequence represents a generation sequence
type Sequence struct {
    ID                 int
    Status             SequenceStatus
    TokenIDs           []int
    LastToken          int
    NumTokens          int
    NumPromptTokens    int
    NumCachedTokens    int
    BlockTable         []int
    Temperature        float32
    MaxTokens          int
    IgnoreEOS          bool
    TopP               float32
    TopK               int
    RepetitionPenalty  float32
    PresencePenalty    float32
    FrequencyPenalty   float32
}

var sequenceCounter int64

// NewSequence creates a new sequence
func NewSequence(tokenIDs []int, params *sampling.SamplingParams) *Sequence {
	id := int(atomic.AddInt64(&sequenceCounter, 1)) - 1
	
	s := &Sequence{
		ID:              id,
		Status:          SequenceStatusWaiting,
		TokenIDs:        make([]int, len(tokenIDs)),
		LastToken:       tokenIDs[len(tokenIDs)-1],
		NumTokens:       len(tokenIDs),
		NumPromptTokens: len(tokenIDs),
        Temperature:     params.Temperature,
        MaxTokens:       params.MaxTokens,
        IgnoreEOS:       params.IgnoreEOS,
        TopP:            params.TopP,
        TopK:            params.TopK,
        RepetitionPenalty: params.RepetitionPenalty,
        PresencePenalty:   params.PresencePenalty,
        FrequencyPenalty:  params.FrequencyPenalty,
    }
	
	// Copy token IDs
	copy(s.TokenIDs, tokenIDs)
	return s
}

// IsFinished checks if the sequence is finished
func (s *Sequence) IsFinished() bool {
	return s.Status == SequenceStatusFinished
}

// NumCompletionTokens returns the number of completion tokens
func (s *Sequence) NumCompletionTokens() int {
	return s.NumTokens - s.NumPromptTokens
}

// PromptTokenIDs returns the prompt token IDs
func (s *Sequence) PromptTokenIDs() []int {
	return s.TokenIDs[:s.NumPromptTokens]
}

// CompletionTokenIDs returns the completion token IDs
func (s *Sequence) CompletionTokenIDs() []int {
	return s.TokenIDs[s.NumPromptTokens:]
}

// AppendToken appends a token to the sequence
func (s *Sequence) AppendToken(tokenID int) {
	s.TokenIDs = append(s.TokenIDs, tokenID)
	s.LastToken = tokenID
	s.NumTokens++
}
