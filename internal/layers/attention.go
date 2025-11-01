package layers

import (
    "fmt"
    "math"

    "github.com/unixsysdev/nano-go-vllm/internal/mathx"
    "github.com/unixsysdev/nano-go-vllm/internal/tensor"
)

// Attention represents an attention layer
type Attention struct {
    numHeads      int
    headDim       int
    scale         float32
    numKVHeads    int
    qProj         *Linear
    kProj         *Linear
    vProj         *Linear
    oProj         *Linear
    rotaryEmbed   *RotaryEmbedding

    // minimal KV cache for single sequence
    kCache [][]float32 // per kv head: [tokens*headDim]
    vCache [][]float32
    cacheLen int
}

// Setters for loading weights from external files
func (a *Attention) SetQWeights(w []float32) error { return a.qProj.LoadWeights(w, nil) }
func (a *Attention) SetKWeights(w []float32) error { return a.kProj.LoadWeights(w, nil) }
func (a *Attention) SetVWeights(w []float32) error { return a.vProj.LoadWeights(w, nil) }
func (a *Attention) SetOWeights(w []float32) error { return a.oProj.LoadWeights(w, nil) }

// NewAttention creates a new attention layer
func NewAttention(hiddenSize, numHeads, numKVHeads, headDim int, maxPosition int, ropeTheta float64, ropeScalingType string, ropeScalingFactor float64) (*Attention, error) {
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	qProj, err := NewLinear(hiddenSize, numHeads*headDim, false)
	if err != nil {
		return nil, fmt.Errorf("failed to create q projection: %v", err)
	}

	kProj, err := NewLinear(hiddenSize, numKVHeads*headDim, false)
	if err != nil {
		return nil, fmt.Errorf("failed to create k projection: %v", err)
	}

	vProj, err := NewLinear(hiddenSize, numKVHeads*headDim, false)
	if err != nil {
		return nil, fmt.Errorf("failed to create v projection: %v", err)
	}

	oProj, err := NewLinear(numHeads*headDim, hiddenSize, false)
	if err != nil {
		return nil, fmt.Errorf("failed to create o projection: %v", err)
	}

    rotaryEmbed, err := NewRotaryEmbedding(headDim, headDim, maxPosition, ropeTheta, ropeScalingType, ropeScalingFactor)
	if err != nil {
		return nil, fmt.Errorf("failed to create rotary embedding: %v", err)
	}

    kCache := make([][]float32, numKVHeads)
    vCache := make([][]float32, numKVHeads)
    return &Attention{
        numHeads:     numHeads,
        headDim:      headDim,
        scale:        scale,
        numKVHeads:   numKVHeads,
        qProj:        qProj,
        kProj:        kProj,
        vProj:        vProj,
        oProj:        oProj,
        rotaryEmbed:  rotaryEmbed,
        kCache:       kCache,
        vCache:       vCache,
        cacheLen:     0,
    }, nil
}

// ResetCache clears KV cache
func (a *Attention) ResetCache() {
    for i := range a.kCache { a.kCache[i] = nil }
    for i := range a.vCache { a.vCache[i] = nil }
    a.cacheLen = 0
}

// Forward performs forward pass with single-seq KV cache.
// input: [T, hidden], positions: [T]
func (a *Attention) Forward(input, positions *tensor.Tensor) (*tensor.Tensor, error) {
    inShape := input.Shape()
    if len(inShape) != 2 {
        return nil, fmt.Errorf("attention input must be 2D [T, hidden]")
    }
    T := inShape[0]

    // Projections
    q, err := a.qProj.Forward(input)
    if err != nil { return nil, fmt.Errorf("q projection failed: %v", err) }
    k, err := a.kProj.Forward(input)
    if err != nil { return nil, fmt.Errorf("k projection failed: %v", err) }
    v, err := a.vProj.Forward(input)
    if err != nil { return nil, fmt.Errorf("v projection failed: %v", err) }

    qData := q.Data().Data().([]float32) // [T, numHeads*headDim]
    kData := k.Data().Data().([]float32) // [T, numKVHeads*headDim]
    vData := v.Data().Data().([]float32) // [T, numKVHeads*headDim]

    headsOut := make([]float32, T*a.numHeads*a.headDim)
    // Append K,V for this block
    prev := a.cacheLen
    for t := 0; t < T; t++ {
        p := prev + t
        if p >= a.rotaryEmbed.maxPosition { p = a.rotaryEmbed.maxPosition - 1 }
        for kv := 0; kv < a.numKVHeads; kv++ {
            kOff := t*a.numKVHeads*a.headDim + kv*a.headDim
            vOff := t*a.numKVHeads*a.headDim + kv*a.headDim
            kVec := make([]float32, a.headDim)
            vVec := make([]float32, a.headDim)
            copy(kVec, kData[kOff:kOff+a.headDim])
            copy(vVec, vData[vOff:vOff+a.headDim])
            a.rotaryEmbed.applyRotary(kVec, p)
            a.kCache[kv] = append(a.kCache[kv], kVec...)
            a.vCache[kv] = append(a.vCache[kv], vVec...)
        }
        a.cacheLen++
    }
    L := a.cacheLen
    for h := 0; h < a.numHeads; h++ {
        kv := h % a.numKVHeads
        // Build Q_h (T x D) with RoPE applied
        qh := make([]float32, T*a.headDim)
        for t := 0; t < T; t++ {
            p := prev + t
            if p >= a.rotaryEmbed.maxPosition { p = a.rotaryEmbed.maxPosition - 1 }
            qOff := t*a.numHeads*a.headDim + h*a.headDim
            vec := make([]float32, a.headDim)
            copy(vec, qData[qOff:qOff+a.headDim])
            a.rotaryEmbed.applyRotary(vec, p)
            copy(qh[t*a.headDim:(t+1)*a.headDim], vec)
        }
        // K_h cache [L x D] and V_h cache [L x D]
        kh := a.kCache[kv]
        vh := a.vCache[kv]
        // scores = qh * kh^T -> [T x L]
        scores := make([]float32, T*L)
        mathx.GemmNT(a.scale, qh, T, a.headDim, kh, L, a.headDim, 0.0, scores)
        // causal softmax row-wise
        for t := 0; t < T; t++ {
            allowed := prev + t + 1
            if allowed > L { allowed = L }
            row := scores[t*L : (t+1)*L]
            for i := allowed; i < L; i++ { row[i] = -1e30 }
            max := row[0]
            for i := 1; i < L; i++ { if row[i] > max { max = row[i] } }
            var sum float32
            for i := 0; i < L; i++ { row[i] = float32(math.Exp(float64(row[i]-max))); sum += row[i] }
            if sum == 0 { sum = 1 }
            inv := 1 / sum
            for i := 0; i < L; i++ { row[i] *= inv }
        }
        // out_h = scores * vh -> [T x D]
        outH := make([]float32, T*a.headDim)
        mathx.GemmNN(1.0, scores, T, L, vh, L, a.headDim, 0.0, outH)
        for t := 0; t < T; t++ {
            outOff := t*a.numHeads*a.headDim + h*a.headDim
            copy(headsOut[outOff:outOff+a.headDim], outH[t*a.headDim:(t+1)*a.headDim])
        }
    }

    // Project concatenated heads
    hs, err := tensor.NewTensor([]int{T, a.numHeads * a.headDim}, tensor.Float32, tensor.CPU)
    if err != nil { return nil, err }
    copy(hs.Data().Data().([]float32), headsOut)
    out, err := a.oProj.Forward(hs)
    if err != nil { return nil, fmt.Errorf("output projection failed: %v", err) }
    return out, nil
}
