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
    RepetitionPenalty float32
    PresencePenalty   float32
    FrequencyPenalty  float32
}

// Sampler represents a token sampler
type Sampler struct{}

// NewSampler creates a new sampler
func NewSampler() *Sampler {
	return &Sampler{}
}

// Sample samples tokens from logits
func (s *Sampler) Sample(logits *tensor.Tensor, temperatures []float32, prevTokens [][]int, params []*SamplingParams) ([]int, error) {
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
        // Penalties / filters configured per sample
        p := DefaultParams
        if params != nil && i < len(params) && params[i] != nil {
            p.TopP = params[i].TopP
            p.TopK = params[i].TopK
            p.RepetitionPenalty = params[i].RepetitionPenalty
            p.PresencePenalty = params[i].PresencePenalty
            p.FrequencyPenalty = params[i].FrequencyPenalty
        }

        // Apply repetition / presence / frequency penalties on logits
        if prevTokens != nil && i < len(prevTokens) {
            applyPenalties(logitSlice, prevTokens[i], p)
        }

        // Top-k filter
        if p.TopK > 0 {
            topKFilter(logitSlice, p.TopK)
        }
        // Softmax -> probs
        probs := softmax(logitSlice)
        // Top-p filter (nucleus)
        if p.TopP > 0 && p.TopP < 1 {
            probs = topPFilter(probs, p.TopP)
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
    RepetitionPenalty float32
    PresencePenalty   float32
    FrequencyPenalty  float32
}{TopP: 0.95, TopK: 50, RepetitionPenalty: 1.0, PresencePenalty: 0.0, FrequencyPenalty: 0.0}

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

func applyPenalties(logits []float32, prev []int, cfg struct{TopP float32; TopK int; RepetitionPenalty, PresencePenalty, FrequencyPenalty float32}) {
    if len(prev) == 0 { return }
    // Count frequencies
    counts := make(map[int]int)
    for _, id := range prev { counts[id]++ }
    for id, c := range counts {
        if id < 0 || id >= len(logits) { continue }
        // repetition penalty: divide or multiply logits
        if cfg.RepetitionPenalty != 0 && cfg.RepetitionPenalty != 1.0 {
            if logits[id] > 0 {
                logits[id] /= cfg.RepetitionPenalty
            } else {
                logits[id] *= cfg.RepetitionPenalty
            }
        }
        // presence penalty shifts logits down if token present
        if cfg.PresencePenalty != 0 {
            logits[id] -= cfg.PresencePenalty
        }
        // frequency penalty scales by count
        if cfg.FrequencyPenalty != 0 {
            logits[id] -= cfg.FrequencyPenalty * float32(c)
        }
    }
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
