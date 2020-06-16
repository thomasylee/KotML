package kotml.math.blas

import kotml.math.Vector

/**
 * `BlasAdapter` provides common BLAS matrix operations in a manner that
 * makes use of the Vector class.
 */
interface BlasAdapter {
    /**
     * Returns the matrix multiplication product of a x b + c.
     * @param a first vector in matrix multiplication
     * @param b second vector in matrix multiplication
     * @param c added vector
     * @return vector containing a x b + c
     */
    fun dgemm(a: Vector, b: Vector, c: Vector?): Vector =
        dgemm(a, b, c, 1.0, 0.0)

    /**
     * Returns the matrix multiplication product of alpha * a x b.
     * @param a first vector in matrix multiplication
     * @param b second vector in matrix multiplication
     * @param alpha scalar multiplier to a x b
     * @return vector containing alpha * a x b
     */
    fun dgemm(a: Vector, b: Vector, alpha: Double = 1.0): Vector =
        dgemm(a, b, null, alpha, 0.0)

    /**
     * Returns the matrix multiplication product of alpha * a x b + beta * c.
     * @param a first vector in matrix multiplication
     * @param b second vector in matrix multiplication
     * @param c added vector
     * @param alpha scalar multiplier to a x b
     * @param beta scalar multiplier to c
     * @return vector containing alpha * a x b + beta * c
     */
    fun dgemm(a: Vector, b: Vector, c: Vector?, alpha: Double, beta: Double): Vector
}
