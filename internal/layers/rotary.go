package layers

import (
    "fmt"
    "math"

    "github.com/unixsysdev/nano-go-vllm/internal/tensor"
)

// RotaryEmbedding represents rotary positional embedding
type RotaryEmbedding struct {
	headDim       int
	rotaryDim     int
	maxPosition   int
	base          float64
	cosCache      []float32
	sinCache      []float32
}

// NewRotaryEmbedding creates a new rotary embedding
func NewRotaryEmbedding(headDim, rotaryDim, maxPosition int, base float64) (*RotaryEmbedding, error) {
	if rotaryDim > headDim {
		return nil, fmt.Errorf("rotary dim cannot exceed head dim")
	}

	// Precompute cos and sin values
	invFreq := make([]float64, rotaryDim/2)
	for i := 0; i < rotaryDim/2; i++ {
		invFreq[i] = 1.0 / math.Pow(base, float64(i*2)/float64(rotaryDim))
	}

	cosCache := make([]float32, maxPosition*rotaryDim/2)
	sinCache := make([]float32, maxPosition*rotaryDim/2)

	for pos := 0; pos < maxPosition; pos++ {
		for i := 0; i < rotaryDim/2; i++ {
			freq := float64(pos) * invFreq[i]
			cosCache[pos*rotaryDim/2+i] = float32(math.Cos(freq))
			sinCache[pos*rotaryDim/2+i] = float32(math.Sin(freq))
		}
	}

	return &RotaryEmbedding{
		headDim:     headDim,
		rotaryDim:   rotaryDim,
		maxPosition: maxPosition,
		base:        base,
		cosCache:    cosCache,
		sinCache:    sinCache,
	}, nil
}

// Forward applies rotary embedding to query and key
func (r *RotaryEmbedding) Forward(positions, query, key *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	queryShape := query.Shape()
	keyShape := key.Shape()
	
	if len(queryShape) != 3 || len(keyShape) != 3 {
		return nil, nil, fmt.Errorf("query and key must be 3D tensors")
	}

	batchSize, seqLen, _ := queryShape[0], queryShape[1], queryShape[2]
	
	queryData := query.Data().Data().([]float32)
	keyData := key.Data().Data().([]float32)
	
	positionData := positions.Data().Data().([]int64)
	
	// Apply rotary embedding
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			pos := int(positionData[i*seqLen+j])
			if pos >= r.maxPosition {
				pos = r.maxPosition - 1
			}
			
			queryOffset := (i*seqLen + j) * r.headDim
			keyOffset := (i*seqLen + j) * r.headDim
			
			// Apply to query
			r.applyRotary(queryData[queryOffset:queryOffset+r.rotaryDim], pos)
			
			// Apply to key
			r.applyRotary(keyData[keyOffset:keyOffset+r.rotaryDim], pos)
		}
	}
	
	// Create output tensors
    queryOut, err := tensor.NewTensor(queryShape, tensor.Float32, tensor.CPU)
    if err != nil {
        return nil, nil, err
    }
    qbuf := queryOut.Data().Data().([]float32)
    copy(qbuf, queryData)
	
    keyOut, err := tensor.NewTensor(keyShape, tensor.Float32, tensor.CPU)
    if err != nil {
        return nil, nil, err
    }
    kbuf := keyOut.Data().Data().([]float32)
    copy(kbuf, keyData)
	
	return queryOut, keyOut, nil
}

// applyRotary applies rotary embedding to a slice of data
func (r *RotaryEmbedding) applyRotary(data []float32, pos int) {
	for i := 0; i < r.rotaryDim/2; i++ {
		idx1 := i * 2
		idx2 := i * 2 + 1
		
		cos := r.cosCache[pos*r.rotaryDim/2+i]
		sin := r.sinCache[pos*r.rotaryDim/2+i]
		
		x1 := data[idx1]
		x2 := data[idx2]
		
		data[idx1] = x1*cos - x2*sin
		data[idx2] = x2*cos + x1*sin
	}
}
