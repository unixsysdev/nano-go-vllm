package layers

import (
    "fmt"
    "math"

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
func NewAttention(hiddenSize, numHeads, numKVHeads, headDim int, maxPosition int) (*Attention, error) {
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

	rotaryEmbed, err := NewRotaryEmbedding(headDim, headDim, maxPosition, 10000.0)
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

    for t := 0; t < T; t++ {
        p := a.cacheLen + t
        if p >= a.rotaryEmbed.maxPosition { p = a.rotaryEmbed.maxPosition - 1 }

        // Append K,V to caches per kv-head
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

        // compute per attention head output
        for h := 0; h < a.numHeads; h++ {
            kv := h % a.numKVHeads
            qOff := t*a.numHeads*a.headDim + h*a.headDim
            qVec := make([]float32, a.headDim)
            copy(qVec, qData[qOff:qOff+a.headDim])
            a.rotaryEmbed.applyRotary(qVec, p)

            L := a.cacheLen
            scores := make([]float32, L)
            kCache := a.kCache[kv]
            for i := 0; i < L; i++ {
                kc := kCache[i*a.headDim : (i+1)*a.headDim]
                // dot in float32
                var s float32
                for d := 0; d < a.headDim; d++ { s += qVec[d] * kc[d] }
                scores[i] = s * a.scale
            }
            // softmax
            max := scores[0]
            for i := 1; i < L; i++ { if scores[i] > max { max = scores[i] } }
            var sum float32
            for i := 0; i < L; i++ {
                scores[i] = float32(math.Exp(float64(scores[i]-max)))
                sum += scores[i]
            }
            if sum == 0 { sum = 1 }
            for i := 0; i < L; i++ { scores[i] /= sum }

            // weighted sum of V
            vCache := a.vCache[kv]
            outHead := make([]float32, a.headDim)
            for i := 0; i < L; i++ {
                w := scores[i]
                vc := vCache[i*a.headDim : (i+1)*a.headDim]
                for d := 0; d < a.headDim; d++ { outHead[d] += w * vc[d] }
            }
            outOff := t*a.numHeads*a.headDim + h*a.headDim
            copy(headsOut[outOff:outOff+a.headDim], outHead)
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
