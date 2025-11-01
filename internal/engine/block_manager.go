package engine

import (
	"crypto/sha1"
	"encoding/binary"
)

// Block represents a KV cache block
type Block struct {
	ID       int
	RefCount int
	Hash     uint32
	Tokens   []int
}

// BlockManager manages KV cache blocks
type BlockManager struct {
	blockSize     int
	blocks        []*Block
	freeBlocks    []int
	hashToBlockID map[uint32]int
}

// NewBlockManager creates a new block manager
func NewBlockManager(numBlocks, blockSize int) *BlockManager {
	blocks := make([]*Block, numBlocks)
	freeBlocks := make([]int, numBlocks)
	
	for i := 0; i < numBlocks; i++ {
		blocks[i] = &Block{
			ID:   i,
			Hash: 0,
		}
		freeBlocks[i] = i
	}
	
	return &BlockManager{
		blockSize:     blockSize,
		blocks:        blocks,
		freeBlocks:    freeBlocks,
		hashToBlockID: make(map[uint32]int),
	}
}

// CanAllocate checks if a sequence can be allocated
func (bm *BlockManager) CanAllocate(seq *Sequence) bool {
	numBlocks := (seq.NumTokens + bm.blockSize - 1) / bm.blockSize
	return len(bm.freeBlocks) >= numBlocks
}

// Allocate allocates blocks for a sequence
func (bm *BlockManager) Allocate(seq *Sequence) {
	numBlocks := (seq.NumTokens + bm.blockSize - 1) / bm.blockSize
	seq.BlockTable = make([]int, numBlocks)
	
	for i := 0; i < numBlocks; i++ {
		start := i * bm.blockSize
		end := start + bm.blockSize
		if end > seq.NumTokens {
			end = seq.NumTokens
		}
		
		tokens := seq.TokenIDs[start:end]
		hash := bm.computeHash(tokens)
		
		// Check if block exists in cache
		if blockID, exists := bm.hashToBlockID[hash]; exists {
			block := bm.blocks[blockID]
			if bm.equalTokens(block.Tokens, tokens) {
				seq.BlockTable[i] = blockID
				block.RefCount++
				seq.NumCachedTokens += len(tokens)
				continue
			}
		}
		
		// Allocate new block
		blockID := bm.freeBlocks[len(bm.freeBlocks)-1]
		bm.freeBlocks = bm.freeBlocks[:len(bm.freeBlocks)-1]
		
		block := bm.blocks[blockID]
		block.RefCount = 1
		block.Hash = hash
		block.Tokens = make([]int, len(tokens))
		copy(block.Tokens, tokens)
		
		seq.BlockTable[i] = blockID
		bm.hashToBlockID[hash] = blockID
	}
}

// Free frees blocks for a sequence
func (bm *BlockManager) Free(seq *Sequence) {
	for _, blockID := range seq.BlockTable {
		block := bm.blocks[blockID]
		block.RefCount--
		
		if block.RefCount == 0 {
			bm.freeBlocks = append(bm.freeBlocks, blockID)
			delete(bm.hashToBlockID, block.Hash)
			block.Hash = 0
			block.Tokens = nil
		}
	}
	
	seq.BlockTable = nil
	seq.NumCachedTokens = 0
}

// CanAppend checks if a sequence can append a token
func (bm *BlockManager) CanAppend(seq *Sequence) bool {
	if len(seq.BlockTable) == 0 {
		return len(bm.freeBlocks) > 0
	}
	
	lastBlockID := seq.BlockTable[len(seq.BlockTable)-1]
	lastBlock := bm.blocks[lastBlockID]
	
	if len(lastBlock.Tokens) < bm.blockSize {
		return true
	}
	
	return len(bm.freeBlocks) > 0
}

// Append appends a token to a sequence
func (bm *BlockManager) Append(seq *Sequence) {
	if len(seq.BlockTable) == 0 {
		// Allocate first block
		blockID := bm.freeBlocks[len(bm.freeBlocks)-1]
		bm.freeBlocks = bm.freeBlocks[:len(bm.freeBlocks)-1]
		
		block := bm.blocks[blockID]
		block.RefCount = 1
		block.Tokens = []int{seq.LastToken}
		
		seq.BlockTable = append(seq.BlockTable, blockID)
		return
	}
	
	lastBlockID := seq.BlockTable[len(seq.BlockTable)-1]
	lastBlock := bm.blocks[lastBlockID]
	
	if len(lastBlock.Tokens) < bm.blockSize {
		// Append to existing block
		lastBlock.Tokens = append(lastBlock.Tokens, seq.LastToken)
		
		// Update hash
		oldHash := lastBlock.Hash
		delete(bm.hashToBlockID, oldHash)
		lastBlock.Hash = bm.computeHash(lastBlock.Tokens)
		bm.hashToBlockID[lastBlock.Hash] = lastBlockID
	} else {
		// Allocate new block
		blockID := bm.freeBlocks[len(bm.freeBlocks)-1]
		bm.freeBlocks = bm.freeBlocks[:len(bm.freeBlocks)-1]
		
		block := bm.blocks[blockID]
		block.RefCount = 1
		block.Tokens = []int{seq.LastToken}
		block.Hash = bm.computeHash(block.Tokens)
		
		seq.BlockTable = append(seq.BlockTable, blockID)
		bm.hashToBlockID[block.Hash] = blockID
	}
}

// computeHash computes hash for tokens
func (bm *BlockManager) computeHash(tokens []int) uint32 {
	h := sha1.New()
	for _, token := range tokens {
		binary.Write(h, binary.LittleEndian, uint32(token))
	}
	hash := h.Sum(nil)
	return binary.LittleEndian.Uint32(hash[:4])
}

// equalTokens checks if two token slices are equal
func (bm *BlockManager) equalTokens(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
