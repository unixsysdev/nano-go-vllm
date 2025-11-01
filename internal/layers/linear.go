package layers

import (
    "fmt"

    "github.com/unixsysdev/nano-go-vllm/internal/tensor"
)

// Linear represents a linear layer
type Linear struct {
	weight *tensor.Tensor
	bias   *tensor.Tensor
}

// NewLinear creates a new linear layer
func NewLinear(inputSize, outputSize int, hasBias bool) (*Linear, error) {
	weight, err := tensor.NewTensor([]int{outputSize, inputSize}, tensor.Float32, tensor.CPU)
	if err != nil {
		return nil, fmt.Errorf("failed to create weight tensor: %v", err)
	}

	var bias *tensor.Tensor
	if hasBias {
		bias, err = tensor.NewTensor([]int{outputSize}, tensor.Float32, tensor.CPU)
		if err != nil {
			return nil, fmt.Errorf("failed to create bias tensor: %v", err)
		}
	}

	return &Linear{
		weight: weight,
		bias:   bias,
	}, nil
}

// Forward performs forward pass
func (l *Linear) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	// input shape: [batchSize, inputSize]
	// weight shape: [outputSize, inputSize]
	// output shape: [batchSize, outputSize]
	
    wT, err := l.weight.Transpose()
    if err != nil {
        return nil, fmt.Errorf("transpose failed: %v", err)
    }
    output, err := input.MatMul(wT)
    if err != nil {
        return nil, fmt.Errorf("matmul failed: %v", err)
    }

	if l.bias != nil {
		output, err = output.Add(l.bias)
		if err != nil {
			return nil, fmt.Errorf("bias add failed: %v", err)
		}
	}

	return output, nil
}

// LoadWeights loads weights from data
func (l *Linear) LoadWeights(weightData, biasData []float32) error {
	// Load weights
    weightDense := l.weight.Data()
    copy(weightDense.Data().([]float32), weightData)

	// Load bias if provided
    if l.bias != nil && biasData != nil {
        biasDense := l.bias.Data()
        copy(biasDense.Data().([]float32), biasData)
    }

	return nil
}
