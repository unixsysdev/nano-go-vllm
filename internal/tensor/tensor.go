package tensor

import (
    "fmt"
    "runtime"
    "sync"

    ggtensor "gorgonia.org/tensor"
)

// Device represents computation device
type Device int

const (
	CPU Device = iota
	GPU
)

// Tensor represents a multi-dimensional array
type Tensor struct {
    data    ggtensor.Tensor
    device  Device
    mu      sync.Mutex
}

// NewTensor creates a new tensor
func NewTensor(shape []int, dtype Dtype, dev Device) (*Tensor, error) {
    data := ggtensor.New(ggtensor.WithShape(shape...), ggtensor.Of(dtype))
	
	if dev == GPU {
		// GPU implementation would go here
		return nil, fmt.Errorf("GPU not implemented yet")
	}
	
	return &Tensor{
		data:   data,
		device: dev,
	}, nil
}

// Data returns the underlying tensor data
func (t *Tensor) Data() ggtensor.Tensor {
    t.mu.Lock()
    defer t.mu.Unlock()
    return t.data
}

// Shape returns the tensor shape
func (t *Tensor) Shape() []int {
    return t.data.Shape()
}

// Dtype returns the tensor data type
func (t *Tensor) Dtype() Dtype {
    return t.data.Dtype()
}

// Device returns the tensor device
func (t *Tensor) Device() Device {
    return t.device
}

// At returns the value at coordinates
func (t *Tensor) At(coord ...int) (interface{}, error) {
    return t.data.At(coord...)
}

// SetAt sets the value at coordinates
func (t *Tensor) SetAt(v interface{}, coord ...int) error {
    return t.data.SetAt(v, coord...)
}

// MatMul performs matrix multiplication
func (t *Tensor) MatMul(other *Tensor) (*Tensor, error) {
	if t.device != other.device {
		return nil, fmt.Errorf("tensor devices must match")
	}
	
    result, err := ggtensor.MatMul(t.data, other.data)
	if err != nil {
		return nil, err
	}
	
	return &Tensor{
		data:   result,
		device: t.device,
	}, nil
}

// Add performs element-wise addition
func (t *Tensor) Add(other *Tensor) (*Tensor, error) {
	if t.device != other.device {
		return nil, fmt.Errorf("tensor devices must match")
	}
	
    result, err := ggtensor.Add(t.data, other.data)
	if err != nil {
		return nil, err
	}
	
	return &Tensor{
		data:   result,
		device: t.device,
	}, nil
}

// Mul performs element-wise multiplication
func (t *Tensor) Mul(other *Tensor) (*Tensor, error) {
	if t.device != other.device {
		return nil, fmt.Errorf("tensor devices must match")
	}
	
    result, err := ggtensor.Mul(t.data, other.data)
	if err != nil {
		return nil, err
	}
	
	return &Tensor{
		data:   result,
		device: t.device,
	}, nil
}

// Reshape reshapes the tensor
func (t *Tensor) Reshape(shape ...int) (*Tensor, error) {
    if err := t.data.Reshape(shape...); err != nil {
        return nil, err
    }
    return &Tensor{data: t.data, device: t.device}, nil
}

// Transpose transposes the tensor
func (t *Tensor) Transpose(axes ...int) (*Tensor, error) {
    // Work on a clone to avoid mutating the original tensor shape/view
    var d ggtensor.Tensor
    if c, ok := t.data.(ggtensor.Cloner); ok {
        d = c.Clone().(ggtensor.Tensor)
    } else {
        d = t.data
    }
    if err := d.T(axes...); err != nil {
        return nil, err
    }
    // Materialize the transpose so subsequent ops see the new layout
    if err := d.Transpose(); err != nil {
        return nil, err
    }
    return &Tensor{data: d, device: t.device}, nil
}

// CopyTo copies tensor to device
func (t *Tensor) CopyTo(dev Device) (*Tensor, error) {
	if t.device == dev {
		return t, nil
	}
	
	// For now, just create a copy on CPU
	// GPU implementation would handle actual device transfer
    data := t.data.Clone()
	
	return &Tensor{
        data:   data.(ggtensor.Tensor),
        device: dev,
    }, nil
}

// Finalizer for GPU memory cleanup
func (t *Tensor) finalize() {
	if t.device == GPU {
		// GPU memory cleanup would go here
		runtime.SetFinalizer(t, nil)
	}
}

// Re-export selected gorgonia.org/tensor types and dtypes for convenience
type (
    Dense = ggtensor.Dense
    Dtype = ggtensor.Dtype
)

var (
    Float32 = ggtensor.Float32
    Float64 = ggtensor.Float64
    Int64   = ggtensor.Int64
    Int32   = ggtensor.Int32
    Int     = ggtensor.Int
    Bool    = ggtensor.Bool
)
