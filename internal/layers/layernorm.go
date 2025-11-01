package layers

import (
    "fmt"
    "math"

    "github.com/unixsysdev/nano-go-vllm/internal/tensor"
)

// RMSNorm represents RMS normalization layer
type RMSNorm struct {
	weight *tensor.Tensor
	eps    float32
}

// NewRMSNorm creates a new RMSNorm layer
func NewRMSNorm(hiddenSize int, eps float32) (*RMSNorm, error) {
    weight, err := tensor.NewTensor([]int{hiddenSize}, tensor.Float32, tensor.CPU)
	if err != nil {
		return nil, fmt.Errorf("failed to create weight tensor: %v", err)
	}

	// Initialize weights to 1
	weightData := make([]float32, hiddenSize)
	for i := range weightData {
		weightData[i] = 1.0
	}
    wbuf := weight.Data().Data().([]float32)
    copy(wbuf, weightData)

	return &RMSNorm{
		weight: weight,
		eps:    eps,
	}, nil
}

// Forward performs forward pass
func (r *RMSNorm) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	shape := input.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("input must be 2D tensor")
	}

	batchSize, hiddenSize := shape[0], shape[1]
	inputData := input.Data().Data().([]float32)
	
	outputData := make([]float32, batchSize*hiddenSize)
	
	for i := 0; i < batchSize; i++ {
		offset := i * hiddenSize
		
		// Compute RMS
		var sumSq float32
		for j := 0; j < hiddenSize; j++ {
			val := inputData[offset+j]
			sumSq += val * val
		}
		rms := float32(math.Sqrt(float64(sumSq/float32(hiddenSize) + r.eps)))
		
		// Normalize and scale
		for j := 0; j < hiddenSize; j++ {
			outputData[offset+j] = inputData[offset+j] / rms
		}
	}
	
	// Apply weight
	weightData := r.weight.Data().Data().([]float32)
	for i := 0; i < batchSize; i++ {
		offset := i * hiddenSize
		for j := 0; j < hiddenSize; j++ {
			outputData[offset+j] *= weightData[j]
		}
	}
	
    output, err := tensor.NewTensor(shape, tensor.Float32, tensor.CPU)
    if err != nil {
        return nil, err
    }
    obuf := output.Data().Data().([]float32)
    copy(obuf, outputData)
	
	return output, nil
}

// LoadWeights loads weights from data
func (r *RMSNorm) LoadWeights(weightData []float32) error {
    weightDense := r.weight.Data()
    copy(weightDense.Data().([]float32), weightData)
    return nil
}
