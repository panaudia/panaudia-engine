package engine

// Channel audibility — decided 2026-07-30 (option D): include/exclude is
// enforced entirely in Layer 2 by handing each listener its own peers
// list, built from a slot-indexed boolean pair matrix. No DSP-layer
// change: the peers slice was always the engine's input, and the encoder
// still applies self-exclusion and moderation mute to whatever it is
// given. (Per-channel gain/attenuation overrides are per-pair *scalars*
// and wait for Phase C's pair-record work — and for the base profile to
// pin how they compose with entity render params, erratum candidate 11.)
//
// The matrix is recomputed whole on any policy change (membership, the
// channel table, entity add/remove) — control-rate work, O(N²·channels)
// on the audio thread — and read per tick to assemble rows. All of it is
// audio-thread-confined.

// markAudibleDirty is called inside change closures.
func (m *Mixer) markAudibleDirty() {
	m.audibleDirty = true
}

// channelAudible: source s is audible to sink k iff they share at least
// one channel that is not muted. Absence from the channel table means an
// unmuted channel (channels exist by being named).
func (m *Mixer) channelAudible(s, k *entity) bool {
	for _, c := range s.params.SourceChannels {
		if m.channelTable[c].Muted {
			continue
		}
		for _, kc := range k.params.SinkChannels {
			if kc == c {
				return true
			}
		}
	}
	return false
}

// recomputeAudible rebuilds the whole pair matrix. Self-pairs are
// computed like any other: self-exclusion is the encoder's own rule (and
// the future hears-self override will live there, not here).
func (m *Mixer) recomputeAudible() {
	m.audibleDirty = false
	for _, k := range m.ents {
		row := m.audible[k.slot]
		for _, s := range m.ents {
			row[s.slot] = m.channelAudible(s, k)
		}
	}
}
