package engine

import (
	"math"
	"sync/atomic"

	"github.com/panaudia/panaudia-engine/buffers"
	"github.com/panaudia/panaudia-engine/inout"
)

// Source is an entity's audio-in half. Write methods run on the caller's
// goroutine (the network receive path) and never block on the render
// loop: decode and jitter-write happen here, at packet cadence, exactly
// as in the incumbent; the mixer tick only ever reads.
//
// The caller delivers packets in arrival order and stops calling Write
// after RemoveSource; a straggler write is harmless (it touches only the
// source's own buffers, never the mixer's structures).
type Source struct {
	m  *Mixer
	id string

	jitter  *buffers.JitterBuffer
	decoder *inout.OpusInputDecoder
	tone    *inout.SineMonoInput

	ring        poseRing
	poseScratch [poseRingSize]poseSample // audio-thread scratch

	samplesWritten uint64 // writer-goroutine only: stream domain of the ring
	loudness       atomic.Uint32
	initialPose    Pose
}

// WriteOpus ingests one packet: pose (nil = pose decimated) and Opus
// payload (empty = pose-only packet), with the gapless packet seq. Decode
// and jitter-write run here on the caller's goroutine.
func (s *Source) WriteOpus(seq uint64, pose *Pose, pkt []byte) {
	if s.tone != nil {
		return
	}
	if len(pkt) > 0 {
		pcm := s.decoder.Decode(pkt)
		s.ingestPCM(pcm)
	}
	if pose != nil {
		s.ring.push(seq, s.samplesWritten, *pose)
	}
}

// WritePCM is the decoded-elsewhere leg: mono float32 at 48 kHz. Same
// pose and seq semantics as WriteOpus.
func (s *Source) WritePCM(seq uint64, pose *Pose, pcm []float32) {
	if s.tone != nil {
		return
	}
	if len(pcm) > 0 {
		s.ingestPCM(pcm)
	}
	if pose != nil {
		s.ring.push(seq, s.samplesWritten, *pose)
	}
}

func (s *Source) ingestPCM(pcm []float32) {
	s.updateLoudness(pcm)
	s.jitter.Write(pcm)
	s.samplesWritten += uint64(len(pcm))
}

// Loudness is the smoothed RMS of ingested audio, for presence reporting.
// Safe from any goroutine.
func (s *Source) Loudness() float32 {
	return math.Float32frombits(s.loudness.Load())
}

const loudnessSmoothing = 0.2 // EMA weight per ingested packet

func (s *Source) updateLoudness(pcm []float32) {
	var sum float64
	for _, v := range pcm {
		sum += float64(v) * float64(v)
	}
	rms := float32(math.Sqrt(sum / float64(len(pcm))))
	prev := math.Float32frombits(s.loudness.Load())
	s.loudness.Store(math.Float32bits(prev + loudnessSmoothing*(rms-prev)))
}

// readFrame fills one 240-sample frame on the audio thread and returns
// the pose of the frame's last sample. The jitter buffer's StreamReadPos
// is the reader's absolute position in the same written-stream domain the
// pose ring is keyed by, so alignment survives startup snaps, laps and
// underruns exactly.
func (s *Source) readFrame(dst []float32) (Pose, bool) {
	if s.tone != nil {
		s.tone.ReadMono(dst)
		return s.initialPose, true
	}
	s.jitter.Read(dst)
	L := uint64(s.jitter.StreamReadPos())
	if L == 0 {
		// Still warm-starting: nothing has played, hold the initial pose.
		return s.initialPose, true
	}
	if p, ok := s.ring.poseAt(L, &s.poseScratch); ok {
		return p, true
	}
	return s.initialPose, true
}
