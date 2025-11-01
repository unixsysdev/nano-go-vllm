package layers

import (
    "fmt"
    "math"

    "github.com/unixsysdev/nano-go-vllm/internal/tensor"
)

// SiluAndMul represents SiLU activation and multiplication
type SiluAndMul struct{}

// NewSiluAndMul creates a new SiluAndMul activation
func NewSiluAndMul() *SiluAndMul {
	return &SiluAndMul{}
}

// Forward performs forward pass
func (s *SiluAndMul) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	shape := input.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("input must be 2D tensor")
	}

	batchSize, hiddenSize := shape[0], shape[1]
	if hiddenSize%2 != 0 {
		return nil, fmt.Errorf("hidden size must be even")
	}

	inputData := input.Data().Data().([]float32)
	outputData := make([]float32, batchSize*hiddenSize/2)
	
	halfSize := hiddenSize / 2
	for i := 0; i < batchSize; i++ {
		inputOffset := i * hiddenSize
		outputOffset := i * halfSize
		
		for j := 0; j < halfSize; j++ {
			// Split into two halves
			x := inputData[inputOffset+j]
			y := inputData[inputOffset+j+halfSize]
			
			// SiLU(x) * y
			silu := x / (1.0 + float32(math.Exp(-float64(x))))
			outputData[outputOffset+j] = silu * y
		}
	}
	
    output, err := tensor.NewTensor([]int{batchSize, halfSize}, tensor.Float32, tensor.CPU)
    if err != nil {
        return nil, err
    }
    buf := output.Data().Data().([]float32)
    copy(buf, outputData)
	
	return output, nil
}
