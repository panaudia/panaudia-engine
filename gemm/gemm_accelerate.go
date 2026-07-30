//go:build darwin && !xsmm

package gemm

// The darwin default — SIGNAL-SHIELDED (2026-07-30). Unshielded,
// cblas_sgemm returns CORRUPTED results — whole-magnitude errors, not
// rounding noise — when a Unix signal is delivered to the calling
// thread mid-computation on Apple silicon (observed Darwin 25.5; AMX
// state apparently not preserved across signal delivery; reported to
// Apple). Diagnosed via TestConcurrentParity flaking under Go: the
// runtime's async-preemption SIGURG is the trigger —
// GODEBUG=asyncpreemptoff=1 goes clean, and the standalone C case in
// accelerate-bug-repro/ is bit-exact with no signals and corrupts in
// seconds under a SIGUSR1 pinger. BLASSetThreading and
// VECLIB_MAXIMUM_THREADS are irrelevant to it.
//
// The fix here is pah_mask_block: signals are blocked (held pending,
// not delivered) for the microseconds of the call — ~375 ns of
// pthread_sigmask overhead on a ~2.7 µs call, still ~3× faster than
// the -tags xsmm path on this hardware. 300 hammered runs of
// TestConcurrentParity green under full Go preemption. The shield
// stays even if Apple fixes the OS bug: it is cheap insurance for
// every unfixed macOS in the field, and TestConcurrentParity remains
// the permanent regression guard.

/*
#cgo CFLAGS: -O2 -DACCELERATE_NEW_LAPACK
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
#include <pthread.h>
#include <signal.h>

// One BLAS thread per call — parallelism belongs to the render workers
// (the production model, and what plan/gemm-backends benchmarked).
// Determinism/perf only — the signal shield, not this, is what makes
// concurrent calls safe (see above).
static void pah_accel_single_threaded(void) {
    BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
}

// Signal shield (2026-07-30): the AMX corruption is triggered by signal
// delivery mid-sgemm, so the call is bracketed by pthread_sigmask —
// blocked signals sit pending (delivery deferred microseconds) instead
// of round-tripping a signal frame through the live AMX state. The
// synchronous fault signals stay unblocked so a genuine crash still
// crashes; the thread's previous mask (Go's) is restored exactly.
static void pah_mask_block(sigset_t *old)
{
    sigset_t all;
    sigfillset(&all);
    sigdelset(&all, SIGSEGV);
    sigdelset(&all, SIGBUS);
    sigdelset(&all, SIGILL);
    sigdelset(&all, SIGFPE);
    sigdelset(&all, SIGTRAP);
    pthread_sigmask(SIG_SETMASK, &all, old);
}

static void pah_encode_fade(int nInputs, int nChannels, int nMaxInputs,
                            int nSamples, const float *inputs,
                            const float *weights, const float *prevWeights,
                            float *output, float *temp)
{
    sigset_t old;
    pah_mask_block(&old);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                nChannels, nSamples, nInputs, 1.0f,
                weights, nMaxInputs, inputs, nSamples, 0.0f,
                output, nSamples);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                nChannels, nSamples, nInputs, 1.0f,
                prevWeights, nMaxInputs, inputs, nSamples, 0.0f,
                temp, nSamples);
    pthread_sigmask(SIG_SETMASK, &old, NULL);

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
