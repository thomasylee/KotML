package kotml.math.blas

import kotml.math.Vector
import org.bytedeco.openblas.global.openblas.CblasNoTrans
import org.bytedeco.openblas.global.openblas.CblasRowMajor
import org.bytedeco.openblas.global.openblas.cblas_dgemm

/**
 * `OpenBlas` uses the JavaCPP OpenBLAS library to perform BLAS operations
 * on a device supported by OpenBLAS.
 */
object OpenBlas : BlasAdapter {
    /**
     * Returns the matrix multiplication product of alpha * a x b + beta * c.
     * @param a first vector in matrix multiplication
     * @param b second vector in matrix multiplication
     * @param c added vector
     * @param alpha scalar multiplier to a x b
     * @param beta scalar multiplier to c
     * @return vector containing alpha * a x b + beta * c
     */
    override fun dgemm(a: Vector, b: Vector, c: Vector?, alpha: Double, beta: Double): Vector {
        val m = a.shape[0]
        val p = b.shape[0]
        val n = b.shape[1]
        val A = a.toDoubleArray()
        val B = b.toDoubleArray()
        val C =
            if (c == null)
                DoubleArray(m * n)
            else
                c.toDoubleArray()

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            m, n, p, alpha, A, p, B, n, beta, C, n)

        var index = 0
        return Vector.ofVectors(m) {
            Vector(n) { C[index++] }
        }
    }
}
