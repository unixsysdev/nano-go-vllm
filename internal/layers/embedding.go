package layers

import (
    "fmt"

    "github.com/unixsysdev/nano-go-vllm/internal/tensor"
)

// Embedding represents an embedding layer
type Embedding struct {
	weight *tensor.Tensor
}

// NewEmbedding creates a new embedding layer
func NewEmbedding(vocabSize, embeddingDim int) (*Embedding, error) {
	weight, err := tensor.NewTensor([]int{vocabSize, embeddingDim}, tensor.Float32, tensor.CPU)
	if err != nil {
		return nil, fmt.Errorf("failed to create weight tensor: %v", err)
	}

	return &Embedding{
		weight: weight,
	}, nil
}

// Forward performs forward pass
func (e *Embedding) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	shape := input.Shape()
	if len(shape) != 1 && len(shape) != 2 {
		return nil, fmt.Errorf("input must be 1D or 2D tensor")
	}

    weightData := e.weight.Data().Data().([]float32)
    // robustly get input IDs as a slice even for length-1 tensors
    var inputIDs []int64
    switch v := input.Data().Data().(type) {
    case []int64:
        inputIDs = v
    case int64:
        inputIDs = []int64{v}
    default:
        return nil, fmt.Errorf("unsupported input dtype for embedding: %T", v)
    }
	
	var batchSize, seqLen int
	if len(shape) == 1 {
		batchSize = 1
		seqLen = shape[0]
	} else {
		batchSize = shape[0]
		seqLen = shape[1]
	}
	
	embeddingDim := e.weight.Shape()[1]
	outputData := make([]float32, batchSize*seqLen*embeddingDim)
	
    for i := 0; i < batchSize; i++ {
        for j := 0; j < seqLen; j++ {
            var tokenID int64
            if len(shape) == 1 {
                tokenID = inputIDs[j]
            } else {
                tokenID = inputIDs[i*seqLen+j]
            }
            if tokenID < 0 || tokenID >= int64(len(weightData)/embeddingDim) {
                return nil, fmt.Errorf("token ID out of range: %d", tokenID)
            }
			
			outputOffset := (i*seqLen + j) * embeddingDim
			weightOffset := tokenID * int64(embeddingDim)
			
			for k := 0; k < embeddingDim; k++ {
				outputData[outputOffset+k] = weightData[weightOffset+int64(k)]
			}
		}
	}
	
	var outputShape []int
	if len(shape) == 1 {
		outputShape = []int{seqLen, embeddingDim}
	} else {
		outputShape = []int{batchSize, seqLen, embeddingDim}
	}
	
    output, err := tensor.NewTensor(outputShape, tensor.Float32, tensor.CPU)
    if err != nil {
        return nil, err
    }
    buf := output.Data().Data().([]float32)
    copy(buf, outputData)
	
	return output, nil
}

// LoadWeights loads weights from data
func (e *Embedding) LoadWeights(weightData []float32) error {
    weightDense := e.weight.Data()
    // copy provided weights into underlying buffer
    copy(weightDense.Data().([]float32), weightData)
    return nil
}
