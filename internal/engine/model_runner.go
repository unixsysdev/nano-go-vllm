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
    if len(seqs) == 0 {
        return nil, nil
    }
    if len(seqs) != 1 {
        return nil, fmt.Errorf("only single-sequence batching supported for now")
    }

    // Prepare input
    inputIDs, positions, err := mr.prepareInput(seqs, isPrefill)
    if err != nil {
        return nil, fmt.Errorf("failed to prepare input: %v", err)
    }

    // Reset KV caches on prefill start
    if isPrefill && seqs[0].NumCachedTokens == 0 {
        mr.model.ResetKVCache()
    }

    // Run model
    logitsAll, err := mr.model.Forward(inputIDs, positions) // [T, vocab]
    if err != nil {
        return nil, fmt.Errorf("model forward failed: %v", err)
    }

    // Take last token logits for sampling: shape [1, vocab]
    shape := logitsAll.Shape()
    if len(shape) != 2 {
        return nil, fmt.Errorf("logits must be 2D")
    }
    T, vocab := shape[0], shape[1]
    last := make([]float32, vocab)
    data := logitsAll.Data().Data().([]float32)
    copy(last, data[(T-1)*vocab:T*vocab])

    lastTensor, err := tensor.NewTensor([]int{1, vocab}, tensor.Float32, tensor.CPU)
    if err != nil { return nil, err }
    copy(lastTensor.Data().Data().([]float32), last)

    // Sample tokens
    temperatures := []float32{seqs[0].Temperature}
    // set sampler defaults from sequence params
    sampling.DefaultParams.TopP = seqs[0].TopP
    sampling.DefaultParams.TopK = seqs[0].TopK
    tokens, err := mr.sampler.Sample(lastTensor, temperatures)
    if err != nil {
        return nil, fmt.Errorf("sampling failed: %v", err)
    }
    return tokens, nil
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
