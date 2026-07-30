//go:build darwin && accelerate && !xsmm

package gemm

// DEMOTED FROM THE DARWIN DEFAULT (2026-07-30, opt-in via -tags
// accelerate): cblas_sgemm returns CORRUPTED results — whole-magnitude
// errors, not rounding noise — when a Unix signal is delivered to the
// calling thread mid-computation on Apple silicon (observed on Darwin
// 25.5; AMX state apparently not preserved across signal delivery).
// Diagnosed via TestConcurrentParity flaking under Go: the Go
// runtime's async-preemption signals (SIGURG) are the trigger —
// GODEBUG=asyncpreemptoff=1 makes 300 hammered runs go clean, and the
// standalone C case in accelerate-bug-repro/ shows the same load
// bit-exact with no signals and corrupting within seconds under a
// SIGUSR1 pinger. The single-threaded force below and
// VECLIB_MAXIMUM_THREADS=1 are irrelevant to it. Any Go process (the
// runtime always signals), and any profiled or timer-driven process,
// is exposed — so this backend is unsafe for the engine on macOS
// regardless of workload. Kept for benchmarking; the darwin default
// is gonum, and -tags xsmm is the fast, correct option (libxsmm's
// AArch64 JIT — no AMX — passes the full hammer under preemption).

/*
#cgo CFLAGS: -O2 -DACCELERATE_NEW_LAPACK
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>

// One BLAS thread per call — parallelism belongs to the render workers
// (the production model, and what plan/gemm-backends benchmarked).
// NOTE (2026-07-30): this does NOT make concurrent calls safe — see the
// demotion note above.
static void pah_accel_single_threaded(void) {
    BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
}

static void pah_encode_fade(int nInputs, int nChannels, int nMaxInputs,
                            int nSamples, const float *inputs,
                            const float *weights, const float *prevWeights,
                            float *output, float *temp)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                nChannels, nSamples, nInputs, 1.0f,
                weights, nMaxInputs, inputs, nSamples, 0.0f,
                output, nSamples);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                nChannels, nSamples, nInputs, 1.0f,
                prevWeights, nMaxInputs, inputs, nSamples, 0.0f,
                temp, nSamples);

    // Crossfade previous->current across the frame. Verbatim from the
    // retired panaudia_utils_internal_encode, including its mixed float/
    // double arithmetic — gemm.go's fadeCombine (the gonum backend's fade)
    // replicates the same expression, so the backends agree bit-exactly.
    float div = (float)(nSamples - 1);
    for (int i = 0; i < nChannels; i++) {
        for (int j = 0; j < nSamples; j++) {
            float v = ((float)j) / div;
            int index = (i * nSamples) + j;
            output[index] = (v * output[index]) + ((1.0 - v) * temp[index]);
        }
    }
}
*/
import "C"

// Backend names the compiled GEMM provider, for startup logs.
const Backend = "accelerate"

func init() {
	C.pah_accel_single_threaded()
}

// Predispatch is a no-op for Accelerate (the libxsmm backend JIT-compiles
// one kernel per (channels, K≤maxSources) here).
func Predispatch(nChannels, maxSources int) {}

// EncodeFade — see the package doc for the contract. One cgo call: the
// GEMM pair and the fade all run in C (the fade loop vectorizes under
// -O2; in Go it measurably slowed the encode path).
func EncodeFade(nInputs, nChannels, nMaxInputs, nSamples int,
	inputs, weights, prevWeights, output, temp []float32) {
	checkShapes(nInputs, nChannels, nMaxInputs, nSamples,
		inputs, weights, prevWeights, output, temp)
	C.pah_encode_fade(C.int(nInputs), C.int(nChannels), C.int(nMaxInputs),
		C.int(nSamples),
		(*C.float)(&inputs[0]),
		(*C.float)(&weights[0]),
		(*C.float)(&prevWeights[0]),
		(*C.float)(&output[0]),
		(*C.float)(&temp[0]))
}
