package safetensors

import (
    "encoding/binary"
    "encoding/json"
    "fmt"
    "io"
    "os"
    "path/filepath"
    "sort"
    "strings"
    "unsafe"
)

// TensorInfo describes a tensor entry in safetensors header
type TensorInfo struct {
    Dtype       string  `json:"dtype"`
    Shape       []int64 `json:"shape"`
    DataOffsets [2]int64 `json:"data_offsets"`
}

// Header is the parsed header map: name -> tensor info
type Header map[string]TensorInfo

// File represents an opened safetensors file
type File struct {
    Path   string
    Header Header
    Data   []byte // full file mapped/loaded into memory for simplicity
    offset int64  // start of data payload (after header)
}

// Open opens a .safetensors file and parses its header
func Open(path string) (*File, error) {
    f, err := os.Open(path)
    if err != nil {
        return nil, err
    }
    defer f.Close()

    var headerLen uint64
    if err := binary.Read(f, binary.LittleEndian, &headerLen); err != nil {
        return nil, fmt.Errorf("read header length: %w", err)
    }

    headerBytes := make([]byte, headerLen)
    if _, err := io.ReadFull(f, headerBytes); err != nil {
        return nil, fmt.Errorf("read header: %w", err)
    }

    // Load entire file (small models acceptable). For large models, prefer mmap.
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, err
    }

    // Parse header JSON
    var raw map[string]json.RawMessage
    if err := json.Unmarshal(headerBytes, &raw); err != nil {
        return nil, fmt.Errorf("parse header json: %w", err)
    }

    header := make(Header)
    for k, v := range raw {
        if k == "__metadata__" {
            continue
        }
        var ti TensorInfo
        if err := json.Unmarshal(v, &ti); err != nil {
            return nil, fmt.Errorf("parse tensor info for %s: %w", k, err)
        }
        header[k] = ti
    }

    return &File{Path: path, Header: header, Data: data, offset: int64(8 + headerLen)}, nil
}

// Multi represents a collection of shard files under a directory
type Multi struct {
    Files []*File
}

// OpenDir loads all .safetensors files from a directory (sorted)
func OpenDir(dir string) (*Multi, error) {
    entries, err := os.ReadDir(dir)
    if err != nil {
        return nil, err
    }
    var paths []string
    for _, e := range entries {
        name := e.Name()
        if strings.HasSuffix(name, ".safetensors") {
            paths = append(paths, filepath.Join(dir, name))
        }
    }
    if len(paths) == 0 {
        return nil, fmt.Errorf("no .safetensors files found in %s", dir)
    }
    sort.Strings(paths)
    var files []*File
    for _, p := range paths {
        f, err := Open(p)
        if err != nil {
            return nil, err
        }
        files = append(files, f)
    }
    return &Multi{Files: files}, nil
}

// Find locates a tensor by name across shards, returning file and info
func (m *Multi) Find(name string) (*File, TensorInfo, bool) {
    for _, f := range m.Files {
        if ti, ok := f.Header[name]; ok {
            return f, ti, true
        }
    }
    return nil, TensorInfo{}, false
}

// ReadRaw returns the raw bytes for a tensor by name
func (f *File) ReadRaw(name string) ([]byte, TensorInfo, error) {
    ti, ok := f.Header[name]
    if !ok {
        return nil, TensorInfo{}, fmt.Errorf("tensor %s not found", name)
    }
    start := f.offset + ti.DataOffsets[0]
    end := f.offset + ti.DataOffsets[1]
    if start < 0 || end < start || end > int64(len(f.Data)) {
        return nil, TensorInfo{}, fmt.Errorf("bad offsets for %s: %v", name, ti.DataOffsets)
    }
    return f.Data[start:end], ti, nil
}

// ReadFloat32 reads and converts tensor to float32 slice (supports F32, F16, BF16)
func (f *File) ReadFloat32(name string) ([]float32, TensorInfo, error) {
    raw, ti, err := f.ReadRaw(name)
    if err != nil {
        return nil, TensorInfo{}, err
    }
    switch strings.ToUpper(ti.Dtype) {
    case "F32":
        if len(raw)%4 != 0 {
            return nil, TensorInfo{}, fmt.Errorf("F32 byte length not multiple of 4: %d", len(raw))
        }
        out := make([]float32, len(raw)/4)
        for i := range out {
            bits := binary.LittleEndian.Uint32(raw[i*4 : i*4+4])
            out[i] = mathFromBits(bits)
        }
        return out, ti, nil
    case "F16":
        if len(raw)%2 != 0 {
            return nil, TensorInfo{}, fmt.Errorf("F16 byte length not multiple of 2: %d", len(raw))
        }
        out := make([]float32, len(raw)/2)
        for i := range out {
            h := binary.LittleEndian.Uint16(raw[i*2 : i*2+2])
            out[i] = float16ToFloat32(h)
        }
        return out, ti, nil
    case "BF16":
        if len(raw)%2 != 0 {
            return nil, TensorInfo{}, fmt.Errorf("BF16 byte length not multiple of 2: %d", len(raw))
        }
        out := make([]float32, len(raw)/2)
        for i := range out {
            h := binary.LittleEndian.Uint16(raw[i*2 : i*2+2])
            out[i] = bfloat16ToFloat32(h)
        }
        return out, ti, nil
    default:
        return nil, TensorInfo{}, fmt.Errorf("unsupported dtype %s for %s", ti.Dtype, name)
    }
}

// mathFromBits converts a uint32 IEEE-754 bits to float32 without importing math to avoid CGO
func mathFromBits(b uint32) float32 { return *(*float32)(unsafe.Pointer(&b)) }

// float16ToFloat32 converts IEEE-754 half-precision to single-precision
func float16ToFloat32(h uint16) float32 {
    // Based on standard conversion logic
    s := uint32(h>>15) & 0x00000001
    e := uint32(h>>10) & 0x0000001F
    f := uint32(h & 0x03FF)
    var out uint32
    if e == 0 {
        if f == 0 {
            out = s << 31
        } else {
            // subnormal
            for (f & 0x0400) == 0 { // normalize
                f <<= 1
                e--
            }
            e++
            f &= 0x03FF
            out = (s << 31) | ((e+112) << 23) | (f << 13)
        }
    } else if e == 31 {
        // Inf/NaN
        out = (s << 31) | 0x7F800000 | (f << 13)
    } else {
        out = (s << 31) | ((e+112) << 23) | (f << 13)
    }
    return mathFromBits(out)
}

// bfloat16ToFloat32 converts BF16 to float32 by placing bf16 as high 16 bits
func bfloat16ToFloat32(h uint16) float32 {
    out := uint32(h) << 16
    return mathFromBits(out)
}
