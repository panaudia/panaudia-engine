package engine

import (
	"time"

	"github.com/panaudia/panaudia-engine/common"
)

// Reverb presets, re-exported so hosts never import the DSP layer.
const (
	ReverbNone       = common.REVERB_NONE
	ReverbTightRoom  = common.REVERB_TIGHT_ROOM
	ReverbSmallRoom  = common.REVERB_SMALL_ROOM
	ReverbMediumRoom = common.REVERB_MEDIUM_ROOM
	ReverbLargeHall  = common.REVERB_LARGE_HALL
	ReverbCathedral  = common.REVERB_CATHEDRAL
)

// FrameSize and SampleRate are the engine's hard invariants (48 kHz,
// 240-sample / 5 ms frames), inherited from the DSP layer.
const (
	FrameSize  = common.FRAME_SIZE
	SampleRate = common.SAMPLE_RATE
)

// Config configures a Mixer. The zero value is not useful — start from
// DefaultConfig.
type Config struct {
	// Order is the ambisonic order of the internal buses (1–3).
	Order int
	// MaxEntities bounds concurrent entities; slot-indexed buffers are
	// sized by it at construction.
	MaxEntities int
	// ReverbPreset is one of the Reverb* constants.
	ReverbPreset int
	// Workers is the number of render worker goroutines.
	Workers int
}

func DefaultConfig() Config {
	return Config{Order: 3, MaxEntities: 64, ReverbPreset: ReverbMediumRoom, Workers: 16}
}

// RenderParams is the render-relevant per-entity state that crosses the
// engine boundary — the shape-(C) per-entity tier. The engine owns the
// defaults (an entity added before its state arrives must render
// sensibly); the host maps its own vocabulary (e.g. the LASA base
// profile's lasa.entity.* keys) onto this struct and sets it wholesale,
// writing DefaultRenderParams fields back on clears. Values are taken
// literally: a nil or empty channel list means member of no channels.
type RenderParams struct {
	// Gain is the source gain multiplier.
	Gain float64
	// Attenuation is the distance-attenuation exponent (0 = same level
	// everywhere, 2 = inverse-square).
	Attenuation float64
	// Size is the source size in metres; 0 is a point source. Plumbed,
	// not yet rendered.
	Size float64
	// Directivity is 0 (omnidirectional) to 1 (maximally directional).
	// Plumbed, not yet rendered.
	Directivity float64
	// SourceChannels and SinkChannels are the entity's channel
	// memberships: a source is audible to a sink iff they share at least
	// one channel that is not muted.
	SourceChannels []string
	SinkChannels   []string
}

// DefaultRenderParams mirrors the LASA base profile's documented
// absence-means-default values; a parity test in the server asserts the
// two cannot drift.
func DefaultRenderParams() RenderParams {
	return RenderParams{
		Gain:           1.0,
		Attenuation:    2.0,
		Size:           0.0,
		Directivity:    0.0,
		SourceChannels: []string{"main"},
		SinkChannels:   []string{"main"},
	}
}

// ChannelParams is the space-scoped per-channel tier. Channels exist by
// being named; an unnamed channel behaves as DefaultChannelParams.
//
// Semantics (base profile §4, pinned 2026-07-30): across a pair's
// shared unmuted channels the highest set gain and lowest set
// attenuation win; the winning gain then MULTIPLIES the entity's
// render gain (unset = identity 1.0), and the winning attenuation
// REPLACES the entity's attenuation only when explicitly set (unset =
// no override). Gain and Attenuation are stored and carried today and
// applied when Phase C's per-pair record lands — at which point this
// struct grows a real "unset" representation for Attenuation (a plain
// float cannot say no-override).
type ChannelParams struct {
	// Muted removes the channel from every audibility intersection.
	Muted       bool
	Gain        float64
	Attenuation float64
}

func DefaultChannelParams() ChannelParams {
	return ChannelParams{Muted: false, Gain: 1.0, Attenuation: 2.0}
}

// SourceConfig configures AddSource.
type SourceConfig struct {
	// InitialPose positions the source until the first pose arrives.
	InitialPose Pose
	// InputChannels is the Opus decode channel count for WriteOpus:
	// 1 (default) or 2 (stereo is downmixed to mono after decode).
	InputChannels int
	// Jitter geometry: the writer's maximum packet duration, the reader
	// cadence, and the warm-start guess for the adaptive late-jitter
	// allowance. Zero values take the defaults (20 ms / 5 ms / 5 ms).
	JitterWriterFrame time.Duration
	JitterReaderFrame time.Duration
	JitterLowInit     time.Duration
	// TestTone, when > 0, makes the source a synthetic sine of that
	// frequency (Hz): Write calls are ignored and the pose stays at
	// InitialPose. For tests and diagnostics only.
	TestTone float64
}

func (c *SourceConfig) applyDefaults() {
	if c.InputChannels == 0 {
		c.InputChannels = 1
	}
	if c.JitterWriterFrame == 0 {
		c.JitterWriterFrame = 20 * time.Millisecond
	}
	if c.JitterReaderFrame == 0 {
		c.JitterReaderFrame = 5 * time.Millisecond
	}
	if c.JitterLowInit == 0 {
		c.JitterLowInit = 5 * time.Millisecond
	}
}
