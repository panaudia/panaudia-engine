// Package gemm is the in-repo home of the ambisonic encode hot path
// (M9.2, plan/m9-saf-exit/plan.md): the crossfaded weights×inputs GEMM
// pair that panaudia_utils_internal_encode provided via SWIG/OpenBLAS.
//
// Contract (identical to internal_encode):
//
//	output[nChannels][nSamples] = fade over the frame from
//	    prevWeights[nChannels][nMaxInputs] · inputs[nInputs][nSamples]
//	to  weights[nChannels][nMaxInputs]     · inputs[nInputs][nSamples]
//
// (row-major, fp32, alpha=1 beta=0, NoTrans; sample 0 = previous weights,
// last sample = current weights; temp is caller-owned scratch of output's
// size so the call allocates nothing).
//
// Backends by build target — see plan/gemm-backends/ for the decision:
//   - darwin: Accelerate cblas_sgemm (gemm_accelerate.go)
//   - linux + xsmm tag: libxsmm pre-dispatched JIT kernels (M9.3)
//   - otherwise: gonum, pure Go, reentrant (gemm_gonum.go)
//
// Every backend is single-threaded per call; parallelism belongs to the
// render workers. All backends are reentrant (Accelerate is safe with
// threading forced off per call shape; gonum is pure Go; libxsmm kernels
// are stateless) — the OpenBLAS shared-scratch race this replaces
// (mixer.go's PANAUDIA_MIXER_GONUM escape hatch) is gone by construction.
package gemm

// checkShapes panics if the buffers cannot hold the requested shape — the
// GEMM writes nChannels×nSamples into output and temp, and a C-side write
// past a short buffer is silent heap corruption (caught live in M9.2: a
// test rig passed a 16-channel mixer a 4-channel reverb output; the C path
// had been overrunning it for months).
func checkShapes(nInputs, nChannels, nMaxInputs, nSamples int,
	inputs, weights, prevWeights, output, temp []float32) {
	if len(output) < nChannels*nSamples || len(temp) < nChannels*nSamples {
		panic("gemm: output/temp shorter than nChannels*nSamples")
	}
	if len(inputs) < nInputs*nSamples {
		panic("gemm: inputs shorter than nInputs*nSamples")
	}
	if len(weights) < (nChannels-1)*nMaxInputs+nInputs ||
		len(prevWeights) < (nChannels-1)*nMaxInputs+nInputs {
		panic("gemm: weights shorter than the nChannels×nMaxInputs layout")
	}
}

// fadeCombine applies the frame crossfade: output = v·output + (1−v)·temp
// with v ramping 0→1 across each channel's nSamples. Bit-exact port of the
// C loop in panaudia_utils_internal_encode — note the C expression mixes
// precisions (`(1.0 - v)` promotes to double, the `v*out` product is a
// float multiply), which the float64/float32 casts here reproduce exactly.
func fadeCombine(output, temp []float32, nChannels, nSamples int) {
	div := float32(nSamples - 1)
	for i := 0; i < nChannels; i++ {
		row := output[i*nSamples : (i+1)*nSamples]
		trow := temp[i*nSamples : (i+1)*nSamples]
		for j := range row {
			v := float32(j) / div
			row[j] = float32(float64(v*row[j]) + (1.0-float64(v))*float64(trow[j]))
		}
	}
}
