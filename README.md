# panaudia-engine

Panaudia's protocol- and transport-blind spatial audio engine:
jitter buffering, Opus decode, spatial
render (bilateral ambisonics with per-ear ITD, parallax and near-field
compensation, reverb), policy, and binaural encode — as a Go library
with no network I/O, no state plane, and no wall-clock.

It is the succession of the panaudia render core: the DSP packages were
**copied, with their tests**, from the panaudia repo (which is frozen
for DSP work) and a fresh orchestration layer was written on top.
`PROVENANCE.md` records the source commit, what was left behind, and
every deliberate divergence since. The design record lives in the
sibling lasa repo: `../lasa/plan/greenfield-standalone.md`.

## Shape

**Layer 1 — DSP** (copied): `ambisonic` (encode, bilateral delay/NFC/
parallax, reverb; ← `gemm`), `binaural` (per-ear SH convolver decode;
← `convolver`, `hrtf`), `buffers` (adaptive JitterBuffer, C-backed
frame buffers), `inout` (Opus decode/encode fronts, dynamics),
`timing`, `common`.

**Layer 2 — orchestration** (fresh): the `engine` package. One `Mixer`
per space; **entities** with a string id and an optional `Source`
(audio in) and/or `Sink` (rendered binaural out). Load-bearing ideas
carried from the incumbent: the In → barrier → Across → Out render
phases, slot allocation, and audio-thread-confined mutation through a
changes queue. New here: dual pose pipelines (source pose rides a
lock-free ring aligned to the audio stream's sample domain; sink pose
is a latest-wins slot read once per tick), and channel-based
audibility via per-listener peer lists.

```go
m, _ := engine.New(engine.DefaultConfig())

src, _ := m.AddSource("alice", engine.SourceConfig{})
snk, _ := m.AddSink("bob", myFrameWriter)          // FrameWriter gets opus frames

src.WriteOpus(seq, &pose, packet)                   // network goroutine; never blocks
snk.SetPose(engine.Pose{Yaw: 0.3})                  // latest-wins, read fresh each tick

go m.Run(timing.NewTicker(5, false))                // or drive m.Process() yourself
```

Poses are metres and radians (x forward, y left, z up, yaw
anticlockwise). The engine's only clocks are packet sequence numbers
and sample indices — no seconds anywhere.

## Hard invariants

- 48 kHz, 240-sample (5 ms) frames, end to end.
- No allocation, locking, or blocking on the audio paths: ingest
  (`Write*`, `SetPose`) is lock-free; render-state mutation happens
  only on the `Process` goroutine.
- `FrameWriter.WriteFrame` must not block; the opus slice is valid
  only during the call.

## Building and testing

Plain `go build ./...` works everywhere — the runtime C (convolver,
frame buffers, libopus via cgo) is in-repo or system-standard. GEMM
backend by build tag:

- **default: gonum** (pure Go, reentrant) — correct everywhere,
  passes the 50-person budget gate.
- **`-tags xsmm`: libxsmm** JIT kernels — production (x86 Linux) and
  the fast option on Apple silicon (AArch64 JIT verified). Needs
  libxsmm built at the pinned SHA (see panaudia's
  `plan/gemm-backends/bench/README.md`).
- **`-tags accelerate`: Apple Accelerate — benchmarking only.**
  Demoted from the darwin default: results are corrupted when a Unix
  signal lands mid-call on Apple silicon (AMX state loss; Go's runtime
  signals constantly). See `gemm/gemm_accelerate.go` and the
  standalone reproducer in `gemm/accelerate-bug-repro/`.

Tests: `go test ./...` (and `-race`). The M-series bilateral anchors,
soak, and the 50-person budget gate are inherited from the source repo
and are the fidelity contract for the copy; performance gates skip
themselves under `-race` and `-short`.
