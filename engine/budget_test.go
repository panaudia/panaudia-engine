package engine

// The Phase A gate: the incumbent's M8 budget validation
// (direct/budget_m8_test.go), ported against the new orchestration. The
// full render loop at target scale must fit the 5 ms frame with
// headroom. Same shape as the original: 50 "people", each a tone source
// AND a binaural-decoding listener (P×(P−1) mix load), decode-then-
// discard outputs so the binaural render path is exercised without opus
// encode — matching the incumbent's ConvolverBinauralNullOutput
// listeners. The assert is a tripwire for order-of-magnitude
// regressions, not a profiler; the gate also runs on the x86 production
// box before Phase A closes.

import (
	"fmt"
	"sort"
	"testing"
	"time"

	"github.com/panaudia/panaudia-engine/binaural"
	"github.com/panaudia/panaudia-engine/inout"
)

// addNullSink mirrors the incumbent budget test's listeners: a sink that
// decodes binaural through the shared convolver set and discards,
// skipping the opus/dynamics stage.
func addNullSink(m *Mixer, id string) error {
	k := &Sink{m: m, id: id,
		nullOut: inout.NewConvolverBinauralNullOutput(
			binaural.NewConvolverDecoder(m.convolverSet), m.channelCount)}
	return m.addSink(id, k)
}

func renderBudget(t *testing.T, people, frames int) (median, p99 time.Duration) {
	t.Helper()

	m, err := New(Config{
		Order:        3,
		MaxEntities:  people + 10,
		ReverbPreset: ReverbMediumRoom,
		Workers:      16,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	// The incumbent placed people on a 0–1 grid in a 40 m space; same
	// geometry in metres.
	for i := 0; i < people; i++ {
		id := fmt.Sprintf("p%d", i)
		if _, err := m.AddSource(id, SourceConfig{
			TestTone:    200.0 + float64(i),
			InitialPose: Pose{X: float64(i%10) * 4.0, Y: float64(i/10) * 4.0, Z: 20.0},
		}); err != nil {
			t.Fatal(err)
		}
		if err := addNullSink(m, id); err != nil {
			t.Fatal(err)
		}
	}
	m.Process() // applies the queued adds, warm-up render

	times := make([]time.Duration, 0, frames)
	for f := 0; f < frames; f++ {
		start := time.Now()
		m.Process()
		times = append(times, time.Since(start))
	}
	sort.Slice(times, func(i, j int) bool { return times[i] < times[j] })
	return times[len(times)/2], times[len(times)*99/100]
}

func TestRenderBudgetAtScale(t *testing.T) {
	if testing.Short() {
		t.Skip("budget run is slow")
	}
	if raceEnabled {
		t.Skip("perf gate is meaningless under the race detector")
	}
	const people = 50
	const frames = 200

	med, p99 := renderBudget(t, people, frames)

	t.Logf("bilateral %d people: median %v p99 %v", people, med, p99)

	if med > 4*time.Millisecond {
		t.Fatalf("median frame %v exceeds the 4 ms budget gate", med)
	}
	if p99 > 5*time.Millisecond {
		t.Fatalf("p99 frame %v exceeds the 5 ms frame budget", p99)
	}
}
