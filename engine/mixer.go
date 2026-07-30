// Package engine is the protocol- and transport-blind spatial audio
// engine: jitter buffering, decode, spatial render, policy, binaural
// encode. It composes the copied DSP packages (Layer 1) under a fresh
// orchestration (Layer 2) — see PROVENANCE.md.
//
// The engine has no wall-clock: its only time bases are packet sequence
// numbers, stream sample indices, and the host's tick. All mutation of
// render state happens on the audio thread (the goroutine calling
// Process), fed through an internal changes queue; ingest and pose
// writes are lock-free and never touch the render structures.
package engine

import (
	"errors"
	"fmt"

	"github.com/google/uuid"

	"github.com/panaudia/panaudia-engine/ambisonic"
	"github.com/panaudia/panaudia-engine/binaural"
	"github.com/panaudia/panaudia-engine/buffers"
	"github.com/panaudia/panaudia-engine/common"
	"github.com/panaudia/panaudia-engine/convolver"
	"github.com/panaudia/panaudia-engine/hrtf"
	"github.com/panaudia/panaudia-engine/inout"

	"sync"
)

var (
	ErrClosed          = errors.New("engine: mixer closed")
	ErrDuplicateSource = errors.New("engine: entity already has a source")
	ErrDuplicateSink   = errors.New("engine: entity already has a sink")
	ErrFull            = errors.New("engine: entity capacity reached")
	ErrUnknownEntity   = errors.New("engine: unknown entity id")
)

// entity is one participant: an optional source (audio in) and an
// optional sink (rendered audio out) sharing a string id, a slot, and an
// ambisonic encoder. Audio-thread-only.
type entity struct {
	id     string
	uid    uuid.UUID // Layer 1's identity token, fresh per entity instance
	slot   int
	enc    *ambisonic.Encoder
	src    *Source
	sink   *Sink
	params RenderParams
	// peers is this listener's channel-filtered source list, assembled
	// each tick from the audibility matrix and handed to EncodePeers —
	// include/exclude policy by construction (channels.go).
	peers []*ambisonic.Encoder
}

// regEntry is the control-plane view of an entity, guarded by Mixer.mu:
// duplicate checks and capacity live here so Add/Remove can answer
// synchronously while the audio-side structures stay thread-confined.
type regEntry struct {
	src  *Source
	sink *Sink
}

// Mixer renders one space. Construct with New; drive with Process (host
// clock) or Run (self-clocked over a Ticker).
type Mixer struct {
	cfg          Config
	channelCount int
	encCfg       common.MixerConfig
	convolverSet *convolver.Set

	mu          sync.Mutex
	reg         map[string]*regEntry
	entityCount int
	closed      bool

	changes chan func()

	// Audio-thread state.
	ents           map[string]*entity
	slots          []*entity
	sourceEntities []*entity
	channelTable   map[string]ChannelParams
	// audible[listenerSlot][sourceSlot]: the channel pair matrix,
	// recomputed whole when audibleDirty (channels.go).
	audible      [][]bool
	audibleDirty bool

	workers   []*worker
	nextWork  int
	wg        sync.WaitGroup
	stop      chan struct{}
	stopOnce  sync.Once
	tickCount int64
}

// New builds a Mixer. Config values are taken literally — start from
// DefaultConfig. The HRTF set, its delay model, and the shared convolver
// filter set are installed here, exactly as the incumbent backend did.
func New(cfg Config) (*Mixer, error) {
	if cfg.Order < 1 || cfg.Order > 3 {
		return nil, errors.New("engine: order must be 1..3")
	}
	if cfg.MaxEntities < 1 {
		return nil, errors.New("engine: MaxEntities must be >= 1")
	}
	if cfg.Workers < 1 {
		return nil, errors.New("engine: Workers must be >= 1")
	}

	m := &Mixer{cfg: cfg}
	m.channelCount = common.ChannelCountForOrder(cfg.Order)
	// Size 1.0: the engine is metres-native, so the copied normalized→
	// metres conversion in SetPosition becomes the identity. The single
	// units conversion point (plan I-D) is Pose.position().
	m.encCfg = common.MixerConfig{
		MaxNodes:     cfg.MaxEntities,
		FrameSize:    common.FRAME_SIZE,
		ChannelCount: m.channelCount,
		Order:        cfg.Order,
		Size:         1.0,
		ReverbPreset: cfg.ReverbPreset,
	}

	set := hrtf.Default()
	if set.Bank(cfg.Order) == nil {
		// The bilateral binaural decode convolves per-order SH filter
		// banks from the HRTF set; an order the set doesn't carry cannot
		// build sinks. The default set is 3rd-order.
		return nil, fmt.Errorf("engine: HRTF set %q carries no order-%d filter bank (orders %v)",
			set.Manifest.Name, cfg.Order, set.Manifest.Orders)
	}
	set.InstallDelayModel()
	m.convolverSet = binaural.NewConvolverSet(set, m.channelCount)

	m.reg = make(map[string]*regEntry)
	m.changes = make(chan func(), 4096)
	m.ents = make(map[string]*entity)
	m.slots = make([]*entity, cfg.MaxEntities)
	m.sourceEntities = make([]*entity, 0, cfg.MaxEntities)
	m.channelTable = make(map[string]ChannelParams)
	m.audible = make([][]bool, cfg.MaxEntities)
	for i := range m.audible {
		m.audible[i] = make([]bool, cfg.MaxEntities)
	}
	m.stop = make(chan struct{})

	m.workers = make([]*worker, cfg.Workers)
	for i := range m.workers {
		m.workers[i] = m.newWorker()
	}
	return m, nil
}

// Close stops Run and the worker pool, then settles all pending
// lifecycle work inline: the changes queue is drained (closure retention
// must not outlive the audio thread — a queued RemoveSink holds cgo
// resources the GC cannot free) and any sinks still attached are
// destroyed. Safe because by contract nobody is driving Process
// concurrently (Run has returned or Process calls have ceased), which
// makes the closing goroutine the audio thread's successor.
func (m *Mixer) Close() {
	m.mu.Lock()
	m.closed = true
	m.mu.Unlock()
	m.stopOnce.Do(func() {
		close(m.stop)
		for _, w := range m.workers {
			close(w.jobs)
		}
		m.drainChanges()
		for _, e := range m.slots {
			if e != nil && e.sink != nil {
				e.sink.destroy()
				e.sink = nil
			}
		}
	})
}

// enqueue hands a mutation to the audio thread. Blocks only if the
// control plane outruns the render loop by the queue depth.
func (m *Mixer) enqueue(f func()) {
	m.changes <- f
}

func (m *Mixer) drainChanges() {
	for {
		select {
		case f := <-m.changes:
			f()
		default:
			return
		}
	}
}

// AddSource attaches a source to entity id, creating the entity if
// needed. The returned Source's Write methods are ready immediately —
// audio written before the next tick simply lands in the jitter buffer.
func (m *Mixer) AddSource(id string, cfg SourceConfig) (*Source, error) {
	cfg.applyDefaults()

	src := &Source{m: m, id: id, initialPose: cfg.InitialPose}
	if cfg.TestTone > 0 {
		src.tone = inout.NewSineMonoInput(cfg.TestTone, common.SAMPLE_RATE, common.FRAME_SIZE)
	} else {
		src.jitter = buffers.NewJitterBuffer(buffers.JitterBufferConfig{
			SampleRate:      common.SAMPLE_RATE,
			NumChannels:     1,
			WriterFrameSize: cfg.JitterWriterFrame,
			ReaderFrameSize: cfg.JitterReaderFrame,
			LowInit:         cfg.JitterLowInit,
		})
		src.decoder = inout.NewOpusInputDecoder(cfg.InputChannels)
	}

	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return nil, ErrClosed
	}
	re := m.reg[id]
	if re != nil && re.src != nil {
		m.mu.Unlock()
		return nil, ErrDuplicateSource
	}
	if re == nil {
		if m.entityCount >= m.cfg.MaxEntities {
			m.mu.Unlock()
			return nil, ErrFull
		}
		re = &regEntry{}
		m.reg[id] = re
		m.entityCount++
	}
	re.src = src
	m.mu.Unlock()

	m.enqueue(func() {
		e := m.ensureEntity(id)
		e.src = src
		e.enc.SetPosition(cfg.InitialPose.position())
		m.markAudibleDirty()
	})
	return src, nil
}

// AddSink attaches a sink to entity id, creating the entity if needed.
// Frames start flowing to w on the next ticks.
func (m *Mixer) AddSink(id string, w FrameWriter) (*Sink, error) {
	k := &Sink{m: m, id: id, w: w,
		out: inout.NewConvolverOpusOutputEncoder(
			binaural.NewConvolverDecoder(m.convolverSet), m.channelCount)}
	if err := m.addSink(id, k); err != nil {
		k.out.BeforeDestroy()
		return nil, err
	}
	return k, nil
}

// addSink is AddSink minus sink construction (shared with the test-only
// null sink, which mirrors the incumbent budget test's decode-then-
// discard listeners).
func (m *Mixer) addSink(id string, k *Sink) error {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return ErrClosed
	}
	re := m.reg[id]
	if re != nil && re.sink != nil {
		m.mu.Unlock()
		return ErrDuplicateSink
	}
	if re == nil {
		if m.entityCount >= m.cfg.MaxEntities {
			m.mu.Unlock()
			return ErrFull
		}
		re = &regEntry{}
		m.reg[id] = re
		m.entityCount++
	}
	re.sink = k
	m.mu.Unlock()

	m.enqueue(func() {
		e := m.ensureEntity(id)
		e.sink = k
		m.markAudibleDirty()
	})
	return nil
}

// RemoveSource detaches entity id's source. Idempotent. The entity
// disappears (and its slot frees) when both halves are gone. The caller
// stops Write calls before removal; a straggler is harmless.
func (m *Mixer) RemoveSource(id string) {
	m.mu.Lock()
	re := m.reg[id]
	if re == nil || re.src == nil {
		m.mu.Unlock()
		return
	}
	re.src = nil
	if re.sink == nil {
		delete(m.reg, id)
		m.entityCount--
	}
	m.mu.Unlock()

	m.enqueue(func() {
		e := m.ents[id]
		if e == nil {
			return
		}
		e.src = nil
		if e.sink == nil {
			m.removeEntity(e)
		}
	})
}

// RemoveSink detaches entity id's sink and releases its cgo resources on
// the audio thread — the funnel lesson (encoder frees only on the mixer
// goroutine, after rendering stops) carried as a rule. Idempotent.
func (m *Mixer) RemoveSink(id string) {
	m.mu.Lock()
	re := m.reg[id]
	if re == nil || re.sink == nil {
		m.mu.Unlock()
		return
	}
	re.sink = nil
	if re.src == nil {
		delete(m.reg, id)
		m.entityCount--
	}
	m.mu.Unlock()

	m.enqueue(func() {
		e := m.ents[id]
		if e == nil || e.sink == nil {
			return
		}
		e.sink.destroy()
		e.sink = nil
		if e.src == nil {
			m.removeEntity(e)
		}
	})
}

// SetRenderParams replaces entity id's render params wholesale (shape
// (C), tier one). The host writes DefaultRenderParams fields back on
// clears; values are applied literally. Channel memberships take effect
// on the next tick via the audibility matrix.
func (m *Mixer) SetRenderParams(id string, p RenderParams) error {
	p.SourceChannels = append([]string(nil), p.SourceChannels...)
	p.SinkChannels = append([]string(nil), p.SinkChannels...)

	m.mu.Lock()
	if m.reg[id] == nil {
		m.mu.Unlock()
		return ErrUnknownEntity
	}
	m.mu.Unlock()

	m.enqueue(func() {
		e := m.ents[id]
		if e == nil {
			return
		}
		e.params = p
		e.enc.SetGain(p.Gain)
		e.enc.SetAttenuation(p.Attenuation)
		m.markAudibleDirty()
	})
	return nil
}

// SetChannel sets the space-scoped params for a named channel (tier
// two). Channels exist by being named; unnamed channels behave as
// DefaultChannelParams. Muted takes effect on the next tick; Gain and
// Attenuation are stored and carried but apply only when the per-pair
// scalar path lands in Phase C (composition semantics now pinned —
// see ChannelParams).
func (m *Mixer) SetChannel(name string, p ChannelParams) {
	m.enqueue(func() {
		m.channelTable[name] = p
		m.markAudibleDirty()
	})
}

// SetEntityMuted is the space-scoped moderation mute: entity id's source
// is silenced in every mix. The entity keeps hearing.
func (m *Mixer) SetEntityMuted(id string, muted bool) error {
	m.mu.Lock()
	if m.reg[id] == nil {
		m.mu.Unlock()
		return ErrUnknownEntity
	}
	m.mu.Unlock()

	m.enqueue(func() {
		if e := m.ents[id]; e != nil {
			e.enc.SetSpaceMuted(muted)
		}
	})
	return nil
}

// ensureEntity creates the audio-side entity on first need: slot,
// identity token, encoder (metres-native, defaults until params arrive).
// Audio thread only.
func (m *Mixer) ensureEntity(id string) *entity {
	if e := m.ents[id]; e != nil {
		return e
	}
	slot := -1
	for i, s := range m.slots {
		if s == nil {
			slot = i
			break
		}
	}
	if slot < 0 {
		// Unreachable: the control-plane capacity check gates admission.
		panic("engine: no free slot")
	}
	e := &entity{id: id, uid: uuid.New(), slot: slot, params: DefaultRenderParams()}
	e.enc = ambisonic.NewEncoder(e.uid, true, e.params.Gain, e.params.Attenuation, m.encCfg, slot)
	e.peers = make([]*ambisonic.Encoder, 0, m.cfg.MaxEntities)
	m.ents[id] = e
	m.slots[slot] = e
	return e
}

// removeEntity frees the slot. The encoder is plain Go memory; slot
// reuse hygiene (delay ramps, NFC state) is the encoders' own
// slot-owner/generation logic. Audio thread only.
func (m *Mixer) removeEntity(e *entity) {
	delete(m.ents, e.id)
	m.slots[e.slot] = nil
	m.markAudibleDirty()
}
