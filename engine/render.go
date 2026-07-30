package engine

import (
	"time"

	"github.com/panaudia/panaudia-engine/ambisonic"
	"github.com/panaudia/panaudia-engine/common"
)

// The render loop keeps the incumbent's load-bearing phase structure:
// In → barrier → Across → barrier → Out. The In/Across barrier is
// correctness, not just scheduling — every listener's fractional-ITD
// reads (and pose-fresh positions) must see completed input rings and
// settled positions. Workers own their mixer scratch (two dry buses +
// reverb), so Across jobs run concurrently without sharing state.

const (
	phaseIn = iota
	phaseAcross
	phaseOut
)

type job struct {
	e     *entity
	phase int
}

type worker struct {
	mixer       *ambisonic.Mixer
	mixerR      *ambisonic.Mixer
	reverbMixer *ambisonic.Mixer
	jobs        chan job
}

func (m *Mixer) newWorker() *worker {
	reverbCfg := m.encCfg
	reverbCfg.ChannelCount = common.REVERB_CHANNELS
	reverbCfg.Order = common.OrderForChannelCount(common.REVERB_CHANNELS)

	w := &worker{
		mixer:       ambisonic.NewMixer(m.encCfg),
		mixerR:      ambisonic.NewMixer(m.encCfg),
		reverbMixer: ambisonic.NewMixer(reverbCfg),
		jobs:        make(chan job, 256),
	}
	go func() {
		for j := range w.jobs {
			m.runJob(w, j)
			m.wg.Done()
		}
	}()
	return w
}

func (m *Mixer) dispatch(e *entity, phase int) {
	m.wg.Add(1)
	m.workers[m.nextWork].jobs <- job{e: e, phase: phase}
	m.nextWork = (m.nextWork + 1) % len(m.workers)
}

func (m *Mixer) runJob(w *worker, j job) {
	switch j.phase {
	case phaseIn:
		j.e.phaseIn()
	case phaseAcross:
		j.e.enc.EncodePeers(j.e.peers, w.mixer, w.mixerR, w.reverbMixer)
	case phaseOut:
		j.e.enc.PostMix()
		j.e.sink.emit(j.e.enc.Output)
	}
}

// phaseIn reads one frame of audio and applies this tick's poses.
// Source role: the frame and the pose of its last sample come from the
// jitter-aligned pipeline together. Sink role: rotation and listener
// position are read once from the latest-wins slot — fresh, and the
// single read means the bilateral encode and the binaural render can
// never disagree within a frame. Runs on a worker; everything touched is
// entity-confined, and the In/Across barrier publishes it.
func (e *entity) phaseIn() {
	if e.src != nil {
		p, ok := e.src.readFrame(e.enc.Input)
		e.enc.PushInputRing()
		if ok {
			e.enc.SetPosition(p.position())
		}
	}
	if e.sink != nil {
		if p, ok := e.sink.pose.load(); ok {
			e.enc.SetRotation(p.rotation())
			e.enc.SetListenerPosition(p.position())
		}
	}
}

// Process runs one 5 ms frame: apply queued changes, then the three
// phases. Call from exactly one goroutine — the audio thread. Never
// blocks on ingest, network, or state.
func (m *Mixer) Process() {
	m.tickCount++
	m.drainChanges()
	if m.audibleDirty {
		m.recomputeAudible()
	}

	m.sourceEntities = m.sourceEntities[:0]
	for _, e := range m.slots {
		if e != nil && e.src != nil {
			m.sourceEntities = append(m.sourceEntities, e)
		}
	}
	// Each listener's peers row: its channel-audible subset of this
	// tick's sources (option D — include/exclude by construction).
	for _, e := range m.slots {
		if e == nil || e.sink == nil {
			continue
		}
		e.peers = e.peers[:0]
		row := m.audible[e.slot]
		for _, s := range m.sourceEntities {
			if row[s.slot] {
				e.peers = append(e.peers, s.enc)
			}
		}
	}

	for _, e := range m.slots {
		if e != nil {
			m.dispatch(e, phaseIn)
		}
	}
	m.wg.Wait()

	for _, e := range m.slots {
		if e != nil && e.sink != nil {
			m.dispatch(e, phaseAcross)
		}
	}
	m.wg.Wait()

	for _, e := range m.slots {
		if e != nil && e.sink != nil {
			m.dispatch(e, phaseOut)
		}
	}
	m.wg.Wait()
}

// Ticker paces Run; satisfied by the copied timing.Ticker. Tick blocks
// until the next frame boundary and returns how long the frame's work
// took.
type Ticker interface {
	Tick() time.Duration
}

// Run drives Process under t until Close. The standalone-server
// convenience; a host with its own clock calls Process directly.
func (m *Mixer) Run(t Ticker) {
	for {
		select {
		case <-m.stop:
			return
		default:
		}
		m.Process()
		t.Tick()
	}
}
