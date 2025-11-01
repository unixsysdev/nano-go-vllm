package engine

import (
    "fmt"

    ggtensor "gorgonia.org/tensor"

    "github.com/unixsysdev/nano-go-vllm/internal/config"
    "github.com/unixsysdev/nano-go-vllm/internal/models"
    "github.com/unixsysdev/nano-go-vllm/internal/sampling"
    "github.com/unixsysdev/nano-go-vllm/internal/tensor"
)

// ModelRunner runs the model inference
type ModelRunner struct {
    config  *config.Config
    model   *models.Qwen3Model
    sampler *sampling.Sampler
}

// NewModelRunner creates a new model runner
func NewModelRunner(cfg *config.Config, model *models.Qwen3Model) (*ModelRunner, error) {
	return &ModelRunner{
		config:  cfg,
		model:   model,
		sampler: sampling.NewSampler(),
	}, nil
}

// Run runs the model on sequences
func (mr *ModelRunner) Run(seqs []*Sequence, isPrefill bool) ([]int, error) {
    if len(seqs) == 0 { return nil, nil }
    // Process each sequence independently (simple batching). Shared model state (KV) is per run.
    out := make([]int, len(seqs))
    for i, s := range seqs {
        // Prepare input for this sequence
        inputIDs, positions, err := mr.prepareInput([]*Sequence{s}, isPrefill)
        if err != nil { return nil, fmt.Errorf("prepare input (seq %d): %v", i, err) }
        // Reset caches at start of prefill for this seq
        if isPrefill && s.NumCachedTokens == 0 { mr.model.ResetKVCache() }
        // Forward
        logitsAll, err := mr.model.Forward(inputIDs, positions)
        if err != nil { return nil, fmt.Errorf("model forward (seq %d): %v", i, err) }
        shape := logitsAll.Shape()
        if len(shape) != 2 { return nil, fmt.Errorf("logits must be 2D") }
        T, vocab := shape[0], shape[1]
        last := make([]float32, vocab)
        data := logitsAll.Data().Data().([]float32)
        copy(last, data[(T-1)*vocab:T*vocab])
        lastTensor, err := tensor.NewTensor([]int{1, vocab}, tensor.Float32, tensor.CPU)
        if err != nil { return nil, err }
        copy(lastTensor.Data().Data().([]float32), last)
        // Sample one
        temps := []float32{s.Temperature}
        prev := [][]int{s.CompletionTokenIDs()}
        params := []*sampling.SamplingParams{{
            TopP: s.TopP, TopK: s.TopK,
            RepetitionPenalty: s.RepetitionPenalty,
            PresencePenalty: s.PresencePenalty,
            FrequencyPenalty: s.FrequencyPenalty,
        }}
        toks, err := mr.sampler.Sample(lastTensor, temps, prev, params)
        if err != nil { return nil, fmt.Errorf("sampling (seq %d): %v", i, err) }
        out[i] = toks[0]
    }
    return out, nil
}

// prepareInput prepares input tensors
func (mr *ModelRunner) prepareInput(seqs []*Sequence, isPrefill bool) (*tensor.Tensor, *tensor.Tensor, error) {
    seq := seqs[0]
    var tokenIDs []int64
    var positions []int64
    if isPrefill {
        start := seq.NumCachedTokens
        for i := start; i < seq.NumTokens; i++ {
            tokenIDs = append(tokenIDs, int64(seq.TokenIDs[i]))
            positions = append(positions, int64(i))
        }
    } else {
        tokenIDs = []int64{int64(seq.LastToken)}
        positions = []int64{int64(seq.NumTokens - 1)}
    }
    inputIDs, err := tensor.NewTensor([]int{len(tokenIDs)}, tensor.Int64, tensor.CPU)
    if err != nil { return nil, nil, err }
    denseIDs := inputIDs.Data().(*ggtensor.Dense)
    for i, v := range tokenIDs { denseIDs.Set(i, v) }
    posT, err := tensor.NewTensor([]int{len(positions)}, tensor.Int64, tensor.CPU)
    if err != nil { return nil, nil, err }
    densePos := posT.Data().(*ggtensor.Dense)
    for i, p := range positions { densePos.Set(i, p) }
    return inputIDs, posT, nil
}
