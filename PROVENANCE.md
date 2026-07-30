# Provenance

This library's Layer 1 (the DSP packages) was **copied, with tests, from the
panaudia repo** — succession, not fork: the source repo is frozen for DSP work
and is not refactored to import this library. Plan and rationale:
`../lasa/plan/greenfield-standalone.md`.

- **Source:** `github.com/panaudia/panaudia` at commit
  `fd76cc86c9b4cdd97058bdadb6f075fe0e3777a2` (clean tree), copied 2026-07-30.
- **Copied as-is** (from `core/<pkg>` to `<pkg>`, import paths rewritten to
  `github.com/panaudia/panaudia-engine/<pkg>`, no other changes):
  `ambisonic`, `binaural`, `convolver`, `hrtf`, `gemm`, `buffers`, `inout`,
  `timing`, `common` — including test data (`ambisonic/reference`,
  `ambisonic/testdata`, `hrtf/sets`) and the in-repo C (`convolver/pahconv.c`,
  `pffft.c`).
- **Deliberately left behind:**
  - `buffers/circular_buffer.go`, `circular_buffer_a.go` + their test —
    legacy pre-JitterBuffer implementations, directroc-only.
  - `inout/encoding.go` + `encoding_test.go`, `io_test.go` — NodeInfo2/3
    state-plane codecs (the old state plane does not cross this seam).
  - `common/control_messages.go`, `common/server_error.go` — server plane.
- **Two extraction shims** (verbatim code moved, not rewritten, so the kept
  files stay byte-identical to the source):
  - `buffers/legacy_iface.go` — `ICircularBuffer`, `CircularBufferStats`,
    `BufferStatsDelegate`, the filling/playing state strings and `defaultVal`,
    which `jitter_buffer.go` shares with the left-behind implementations.
  - `inout/f32codec.go` — the zero-copy `Encodef32`/`Decodef32` byte views
    from `encoding.go`, the only part of it the audio path uses.
- **Verified at copy time:** `go build ./...`, `go vet ./...` (pre-existing
  unsafe.Pointer notes only), `go test ./...` and `go test -race ./...` all
  green; the ambisonic suite runs the same 44 tests with the same single
  pre-existing skip (`TestMultiMix`) as the source repo at the pinned commit.
  The `xsmm` build tag (x86 Linux libxsmm backend) is carried but was not
  exercised on this (darwin) machine.

Layer 2 (entity model, phase-barrier render loop, pose store, pair matrix) is
written fresh in this repo (the `engine` package) and is not a copy of
`core/space`.

## Divergences from the copy

Deliberate Layer 1 changes made after the copy, each agreed with Paul and
listed here (the plan's amended Phase A rule: small changes permitted,
every one surfaced and recorded):

1. **`buffers/jitter_stream_pos.go`** (new file, 2026-07-30) —
   `JitterBuffer.StreamReadPos()`: exposes the reader's absolute position
   in the written sample stream, the alignment key for the engine's
   source-pose store. No copied file touched.
2. **Listener-position split** (2026-07-30) — `ambisonic/
   listener_position.go` (new) plus minimal edits to `encoder.go` (two
   struct fields, two listener-argument call sites) and `parallax.go`
   (one call site): `SetListenerPosition`/`ListenerPositionMeters()` let
   the listener role render from a different position than the source
   role (sink pose fresh, source pose jitter-aligned). Fallback preserves
   historical behaviour exactly when the setter is never called; the
   copied test suites pass unchanged.
3. **GEMM: Accelerate signal shield + xsmm on darwin** (2026-07-30) —
   `TestConcurrentParity` (flaking in the copied suite, inherited from
   the source repo) was diagnosed as Accelerate's `cblas_sgemm`
   returning corrupted results when a signal is delivered mid-call on
   Apple silicon (Darwin 25.5 — AMX state seemingly unpreserved across
   signal delivery; Go's async-preemption SIGURG is the trigger,
   `asyncpreemptoff=1` goes clean, and
   `gemm/accelerate-bug-repro/repro.c` reproduces it in pure C with a
   SIGUSR1 pinger while the no-signal load is bit-exact; reported to
   Apple). Changes kept: **(a)** `gemm_accelerate.go` brackets the
   call with `pthread_sigmask` (synchronous fault signals stay
   unblocked; ~375 ns/call) — with the shield, 300 hammered
   concurrent-parity runs pass and Accelerate **remains the darwin
   default**; **(b)** **libxsmm extended to darwin** (`(linux ||
   darwin) && xsmm`, plus the same widening of `xsmm_test.go`) — its
   AArch64 JIT passes the hammer, the JIT-serves gate, and the budget
   gate on macOS at the pinned SHA. (An interim gonum-default state
   existed for part of 2026-07-30 and was superseded by the shield.)
