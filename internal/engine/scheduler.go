package engine

import (
    "container/list"

    "github.com/unixsysdev/nano-go-vllm/internal/config"
)

// Scheduler manages sequence scheduling
type Scheduler struct {
	maxNumSeqs           int
	maxNumBatchedTokens  int
	eosTokenID           int
	blockManager         *BlockManager
	waitingQueue         *list.List
	runningQueue         *list.List
}

// NewScheduler creates a new scheduler
func NewScheduler(config *config.Config) *Scheduler {
	return &Scheduler{
		maxNumSeqs:          config.MaxNumSeqs,
		maxNumBatchedTokens: config.MaxNumBatchedTokens,
		eosTokenID:          config.EOSTokenID,
		blockManager:        NewBlockManager(config.NumKVCacheBlocks, config.KVCacheBlockSize),
		waitingQueue:        list.New(),
		runningQueue:        list.New(),
	}
}

// Add adds a sequence to the waiting queue
func (s *Scheduler) Add(seq *Sequence) {
	s.waitingQueue.PushBack(seq)
}

// Schedule schedules sequences for execution
func (s *Scheduler) Schedule() ([]*Sequence, bool) {
	scheduled := make([]*Sequence, 0)
	
	// Try to schedule from waiting queue (prefill)
	for s.waitingQueue.Len() > 0 && len(scheduled) < s.maxNumSeqs {
		elem := s.waitingQueue.Front()
		seq := elem.Value.(*Sequence)
		
		if s.canSchedule(seq, len(scheduled)) {
			s.waitingQueue.Remove(elem)
			s.blockManager.Allocate(seq)
			seq.Status = SequenceStatusRunning
			s.runningQueue.PushBack(seq)
			scheduled = append(scheduled, seq)
		} else {
			break
		}
	}
	
	if len(scheduled) > 0 {
		return scheduled, true // prefill
	}
	
	// Schedule from running queue (decode)
	for elem := s.runningQueue.Front(); elem != nil && len(scheduled) < s.maxNumSeqs; {
		seq := elem.Value.(*Sequence)
		
		if s.blockManager.CanAppend(seq) {
			s.blockManager.Append(seq)
			scheduled = append(scheduled, seq)
			elem = elem.Next()
		} else {
			// Preempt this sequence
			s.runningQueue.Remove(elem)
			s.blockManager.Free(seq)
			seq.Status = SequenceStatusWaiting
			s.waitingQueue.PushFront(seq)
			elem = s.runningQueue.Front()
		}
	}
	
	return scheduled, false // decode
}

// PostProcess processes the output tokens
func (s *Scheduler) PostProcess(seqs []*Sequence, tokenIDs []int) []bool {
	finished := make([]bool, len(seqs))
	
	for i, seq := range seqs {
		seq.AppendToken(tokenIDs[i])
		
		// Check if finished
		if (!seq.IgnoreEOS && tokenIDs[i] == s.eosTokenID) || 
		   seq.NumCompletionTokens() >= seq.MaxTokens {
			seq.Status = SequenceStatusFinished
			s.blockManager.Free(seq)
			
			// Remove from running queue
			for e := s.runningQueue.Front(); e != nil; e = e.Next() {
				if e.Value.(*Sequence).ID == seq.ID {
					s.runningQueue.Remove(e)
					break
				}
			}
			
			finished[i] = true
		}
	}
	
	return finished
}

// IsFinished checks if all sequences are finished
func (s *Scheduler) IsFinished() bool {
	return s.waitingQueue.Len() == 0 && s.runningQueue.Len() == 0
}

// canSchedule checks if a sequence can be scheduled
func (s *Scheduler) canSchedule(seq *Sequence, currentBatchSize int) bool {
	if currentBatchSize >= s.maxNumSeqs {
		return false
	}
	
	tokensNeeded := seq.NumTokens - seq.NumCachedTokens
	if tokensNeeded > s.maxNumBatchedTokens {
		return false
	}
	
	return s.blockManager.CanAllocate(seq)
}
