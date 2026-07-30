//go:build darwin

package gemm

/*
#cgo CFLAGS: -O2 -DACCELERATE_NEW_LAPACK
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>

// One BLAS thread per call — parallelism belongs to the render workers
// (the production model, and what plan/gemm-backends benchmarked).
// Accelerate's internal threading also partitions the GEMM load-
// dependently, changing float summation order run to run — caught by
// TestConcurrentParity's exact-equality check under full-suite load.
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
