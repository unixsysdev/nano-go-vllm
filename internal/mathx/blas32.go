package mathx

import (
    "gonum.org/v1/gonum/blas"
    b32 "gonum.org/v1/gonum/blas/blas32"
)

// GemmNN computes C = alpha*A*B + beta*C for row-major float32 matrices.
// A is (ar x ac), B is (br x bc) where ac==br. C is (ar x bc).
func GemmNN(alpha float32, A []float32, ar, ac int, B []float32, br, bc int, beta float32, C []float32) {
    // Wrap as blas32.General with row-major stride=cols
    a := b32.General{Rows: ar, Cols: ac, Data: A, Stride: ac}
    b := b32.General{Rows: br, Cols: bc, Data: B, Stride: bc}
    c := b32.General{Rows: ar, Cols: bc, Data: C, Stride: bc}
    b32.Gemm(blas.NoTrans, blas.NoTrans, alpha, a, b, beta, c)
}

// GemmNT computes C = alpha*A*B^T + beta*C for row-major float32 matrices.
func GemmNT(alpha float32, A []float32, ar, ac int, B []float32, br, bc int, beta float32, C []float32) {
    // Compute C = alpha*A*B^T + beta*C
    // A: ar x ac, B: br x bc, C: ar x br
    a := b32.General{Rows: ar, Cols: ac, Data: A, Stride: ac}
    b := b32.General{Rows: br, Cols: bc, Data: B, Stride: bc}
    c := b32.General{Rows: ar, Cols: br, Data: C, Stride: br}
    b32.Gemm(blas.NoTrans, blas.Trans, alpha, a, b, beta, c)
}
