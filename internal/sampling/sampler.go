package sampling

import (
    "fmt"
    "math"
    "math/rand"
    "sort"

    "github.com/unixsysdev/nano-go-vllm/internal/tensor"
)

// SamplingParams holds sampling parameters
type SamplingParams struct {
    Temperature float32
    MaxTokens   int
    IgnoreEOS   bool
    TopP        float32
    TopK        int
}

// Sampler represents a token sampler
type Sampler struct{}

// NewSampler creates a new sampler
func NewSampler() *Sampler {
	return &Sampler{}
}

// Sample samples tokens from logits
func (s *Sampler) Sample(logits *tensor.Tensor, temperatures []float32) ([]int, error) {
    shape := logits.Shape()
    if len(shape) != 2 {
        return nil, fmt.Errorf("logits must be 2D tensor")
    }

	batchSize, vocabSize := shape[0], shape[1]
	logitsData := logits.Data().Data().([]float32)
	
    tokens := make([]int, batchSize)
    
    for i := 0; i < batchSize; i++ {
        offset := i * vocabSize
        // Work on a copy to avoid mutating upstream values
        logitSlice := make([]float32, vocabSize)
        copy(logitSlice, logitsData[offset:offset+vocabSize])
        
        // Apply temperature
        temp := temperatures[i]
        if temp > 0 {
            for j := range logitSlice {
                logitSlice[j] /= temp
            }
        }
        // Top-k filter
        if DefaultParams.TopK > 0 {
            topKFilter(logitSlice, DefaultParams.TopK)
        }
        // Softmax -> probs
        probs := softmax(logitSlice)
        // Top-p filter (nucleus)
        if DefaultParams.TopP > 0 && DefaultParams.TopP < 1 {
            probs = topPFilter(probs, DefaultParams.TopP)
        }
        // Sample token
        tokens[i] = sampleFromProbs(probs)
    }
    
    return tokens, nil
}

// DefaultParams is used for Sampler when not provided per-sequence (simple path)
var DefaultParams = struct{
    TopP float32
    TopK int
}{TopP: 0.95, TopK: 50}

func topKFilter(logits []float32, k int) {
    if k <= 0 || k >= len(logits) { return }
    // Find kth largest threshold
    // Copy indices and sort by logit desc
    idx := make([]int, len(logits))
    for i := range idx { idx[i] = i }
    sort.Slice(idx, func(i, j int) bool { return logits[idx[i]] > logits[idx[j]] })
    thresh := logits[idx[k-1]]
    for i := range logits {
        if logits[i] < thresh {
            logits[i] = -1e30
        }
    }
}

func topPFilter(probs []float32, p float32) []float32 {
    // Sort indices by prob desc
    idx := make([]int, len(probs))
    for i := range idx { idx[i] = i }
    sort.Slice(idx, func(i, j int) bool { return probs[idx[i]] > probs[idx[j]] })
    var cum float32
    cutoff := len(probs)
    for i, id := range idx {
        cum += probs[id]
        if cum >= p { cutoff = i+1; break }
    }
    // Zero out tail and renormalize
    var sum float32
    for i, id := range idx {
        if i >= cutoff { probs[id] = 0 } else { sum += probs[id] }
    }
    if sum > 0 {
        inv := 1 / sum
        for i := range probs { probs[i] *= inv }
    }
    return probs
}

// softmax computes softmax probabilities
func softmax(logits []float32) []float32 {
	// Find max for numerical stability
	max := logits[0]
	for _, v := range logits {
		if v > max {
			max = v
		}
	}
	
    // Compute exp and sum
    expVals := make([]float32, len(logits))
    var sum float32
    for i, v := range logits {
        val := float32(math.Exp(float64(v - max)))
        expVals[i] = val
        sum += val
    }
	
	// Normalize
	for i := range expVals {
		expVals[i] /= sum
	}
	
	return expVals
}

// sampleFromProbs samples from probability distribution
func sampleFromProbs(probs []float32) int {
	// Generate random number
	r := rand.Float32()
	
	// Find the token
	var cumSum float32
	for i, p := range probs {
		cumSum += p
		if r < cumSum {
			return i
		}
	}
	
	return len(probs) - 1
}
