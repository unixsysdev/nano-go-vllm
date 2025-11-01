package config

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// Config holds the configuration for the LLM engine
type Config struct {
	ModelPath                string  `json:"model_path"`
	MaxNumBatchedTokens     int     `json:"max_num_batched_tokens"`
	MaxNumSeqs              int     `json:"max_num_seqs"`
	MaxModelLen             int     `json:"max_model_len"`
	GPUMemoryUtilization    float64 `json:"gpu_memory_utilization"`
	TensorParallelSize      int     `json:"tensor_parallel_size"`
	EnforceEager            bool    `json:"enforce_eager"`
	KVCacheBlockSize        int     `json:"kvcache_block_size"`
	NumKVCacheBlocks        int     `json:"num_kvcache_blocks"`
	
	// Model-specific config
	VocabSize               int     `json:"vocab_size"`
	HiddenSize              int     `json:"hidden_size"`
	NumHiddenLayers         int     `json:"num_hidden_layers"`
	NumAttentionHeads      int     `json:"num_attention_heads"`
	NumKeyValueHeads       int     `json:"num_key_value_heads"`
	IntermediateSize        int     `json:"intermediate_size"`
	HiddenAct               string  `json:"hidden_act"`
	MaxPositionEmbeddings   int     `json:"max_position_embeddings"`
	RMSNormEps              float64 `json:"rms_norm_eps"`
    HeadDim                 int     `json:"head_dim"`
    EOSTokenID              int     `json:"eos_token_id"`
    RoPETheta               float64 `json:"rope_theta"`
    RopeScalingType         string  `json:"-"`
    RopeScalingFactor       float64 `json:"-"`
}

// LoadConfig loads configuration from model path
func LoadConfig(modelPath string, opts ...Option) (*Config, error) {
    cfg := &Config{
        ModelPath:             modelPath,
        MaxNumBatchedTokens:   16384,
        MaxNumSeqs:            512,
        MaxModelLen:           4096,
        GPUMemoryUtilization:  0.9,
        TensorParallelSize:    1,
        EnforceEager:          false,
        KVCacheBlockSize:      256,
        NumKVCacheBlocks:      -1,
    }

	for _, opt := range opts {
		opt(cfg)
	}

	// Validate model path
	if _, err := os.Stat(cfg.ModelPath); os.IsNotExist(err) {
		return nil, err
	}

	// Load model-specific config
	modelConfigPath := filepath.Join(cfg.ModelPath, "config.json")
    if data, err := os.ReadFile(modelConfigPath); err == nil {
        var modelConfig map[string]interface{}
        if err := json.Unmarshal(data, &modelConfig); err != nil {
            return nil, err
        }
		
		// Extract relevant fields
		if v, ok := modelConfig["vocab_size"].(float64); ok {
			cfg.VocabSize = int(v)
		}
		if v, ok := modelConfig["hidden_size"].(float64); ok {
			cfg.HiddenSize = int(v)
		}
		if v, ok := modelConfig["num_hidden_layers"].(float64); ok {
			cfg.NumHiddenLayers = int(v)
		}
		if v, ok := modelConfig["num_attention_heads"].(float64); ok {
			cfg.NumAttentionHeads = int(v)
		}
		if v, ok := modelConfig["num_key_value_heads"].(float64); ok {
			cfg.NumKeyValueHeads = int(v)
		} else {
			cfg.NumKeyValueHeads = cfg.NumAttentionHeads
		}
		if v, ok := modelConfig["intermediate_size"].(float64); ok {
			cfg.IntermediateSize = int(v)
		}
		if v, ok := modelConfig["hidden_act"].(string); ok {
			cfg.HiddenAct = v
		}
		if v, ok := modelConfig["max_position_embeddings"].(float64); ok {
			cfg.MaxPositionEmbeddings = int(v)
		}
		if v, ok := modelConfig["rms_norm_eps"].(float64); ok {
			cfg.RMSNormEps = v
		}
        if v, ok := modelConfig["head_dim"].(float64); ok {
            cfg.HeadDim = int(v)
        } else {
            cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
        }
        if v, ok := modelConfig["rope_theta"].(float64); ok {
            cfg.RoPETheta = v
        } else {
            cfg.RoPETheta = 10000.0
        }
        if rs, ok := modelConfig["rope_scaling"].(map[string]interface{}); ok {
            if t, ok := rs["type"].(string); ok { cfg.RopeScalingType = t }
            if f, ok := rs["factor"].(float64); ok { cfg.RopeScalingFactor = f }
        }
    }

    // If NumKVCacheBlocks not provided, derive a conservative default
    if cfg.NumKVCacheBlocks <= 0 {
        blocksPerSeq := (cfg.MaxModelLen + cfg.KVCacheBlockSize - 1) / cfg.KVCacheBlockSize
        if blocksPerSeq < 1 { blocksPerSeq = 1 }
        // Assume at least single seq; our runner currently enforces single sequence
        cfg.NumKVCacheBlocks = blocksPerSeq * 2
    }

    return cfg, nil
}

// Option is a function that modifies the config
type Option func(*Config)

// WithMaxNumBatchedTokens sets the maximum number of batched tokens
func WithMaxNumBatchedTokens(v int) Option {
	return func(c *Config) { c.MaxNumBatchedTokens = v }
}

// WithMaxNumSeqs sets the maximum number of sequences
func WithMaxNumSeqs(v int) Option {
	return func(c *Config) { c.MaxNumSeqs = v }
}

// WithMaxModelLen sets the maximum model length
func WithMaxModelLen(v int) Option {
	return func(c *Config) { c.MaxModelLen = v }
}

// WithGPUMemoryUtilization sets the GPU memory utilization
func WithGPUMemoryUtilization(v float64) Option {
	return func(c *Config) { c.GPUMemoryUtilization = v }
}

// WithTensorParallelSize sets the tensor parallel size
func WithTensorParallelSize(v int) Option {
	return func(c *Config) { c.TensorParallelSize = v }
}

// WithEnforceEager sets whether to enforce eager execution
func WithEnforceEager(v bool) Option {
	return func(c *Config) { c.EnforceEager = v }
}

// WithKVCacheBlockSize sets the KV cache block size
func WithKVCacheBlockSize(v int) Option {
	return func(c *Config) { c.KVCacheBlockSize = v }
}

// WithNumKVCacheBlocks sets the number of KV cache blocks
func WithNumKVCacheBlocks(v int) Option {
	return func(c *Config) { c.NumKVCacheBlocks = v }
}
