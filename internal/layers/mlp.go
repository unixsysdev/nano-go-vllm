package layers

import (
    "fmt"

    "github.com/unixsysdev/nano-go-vllm/internal/tensor"
)

// MLP represents a multi-layer perceptron
type MLP struct {
    gateUpProj *Linear
    downProj   *Linear
    actFn      *SiluAndMul
}

// NewMLP creates a new MLP layer
func NewMLP(hiddenSize, intermediateSize int, hiddenAct string) (*MLP, error) {
	if hiddenAct != "silu" {
		return nil, fmt.Errorf("only silu activation is supported")
	}

	gateUpProj, err := NewLinear(hiddenSize, intermediateSize*2, false)
	if err != nil {
		return nil, fmt.Errorf("failed to create gate_up projection: %v", err)
	}

	downProj, err := NewLinear(intermediateSize, hiddenSize, false)
	if err != nil {
		return nil, fmt.Errorf("failed to create down projection: %v", err)
	}

	return &MLP{
		gateUpProj: gateUpProj,
		downProj:   downProj,
		actFn:      NewSiluAndMul(),
	}, nil
}

// Forward performs forward pass
func (m *MLP) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	// Gate and up projection
	gateUp, err := m.gateUpProj.Forward(input)
	if err != nil {
		return nil, fmt.Errorf("gate_up projection failed: %v", err)
	}

	// Activation
	activated, err := m.actFn.Forward(gateUp)
	if err != nil {
		return nil, fmt.Errorf("activation failed: %v", err)
	}

	// Down projection
	output, err := m.downProj.Forward(activated)
	if err != nil {
		return nil, fmt.Errorf("down projection failed: %v", err)
	}

	return output, nil
}

// LoadWeights loads weights from data
func (m *MLP) LoadWeights(gateUpWeight, downUpWeight []float32) error {
    if err := m.gateUpProj.LoadWeights(gateUpWeight, nil); err != nil {
        return fmt.Errorf("failed to load gate_up weights: %v", err)
    }
    
    if err := m.downProj.LoadWeights(downUpWeight, nil); err != nil {
        return fmt.Errorf("failed to load down weights: %v", err)
    }
    
    return nil
}

// SetGateUpWeights loads fused gate_up projection weights
func (m *MLP) SetGateUpWeights(w []float32) error { return m.gateUpProj.LoadWeights(w, nil) }

// SetDownWeights loads down projection weights
func (m *MLP) SetDownWeights(w []float32) error { return m.downProj.LoadWeights(w, nil) }
