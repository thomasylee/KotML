package kotml.math.blas

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.cudaDataType.CUDA_R_64F
import jcuda.jcublas.JCublas2
import jcuda.jcublas.JCublas2.cublasCreate
import jcuda.jcublas.JCublas2.cublasDestroy
import jcuda.jcublas.JCublas2.cublasGetVector
import jcuda.jcublas.JCublas2.cublasSetVector
import jcuda.jcublas.cublasHandle
import jcuda.jcublas.cublasOperation
import jcuda.runtime.JCuda.cudaDeviceSynchronize
import jcuda.runtime.JCuda.cudaFree
import jcuda.runtime.JCuda.cudaMalloc
import kotml.math.Vector

/**
 * `CuBlas` uses the JCuda and JCublas libraries to perform BLAS operations
 * on a CUDA-supported GPU.
 */
object CuBlas : BlasAdapter {
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
        val n = if (a.dimensions == 1) 1 else b.shape[1]
        val handle = cublasHandle()
        cublasCreate(handle)

        val ptrAlpha = Pointer.to(DoubleArray(1) { alpha })
        val ptrBeta = Pointer.to(DoubleArray(1) { beta })

        val A = a.toDoubleArray()
        val B = b.toDoubleArray()
        val C =
            if (c == null)
                DoubleArray(m * n)
            else
                c.toDoubleArray()

        val ptrA = Pointer()
        val ptrB = Pointer()
        val ptrC = Pointer()
        cudaMalloc(ptrA, m * p * Sizeof.DOUBLE.toLong())
        cudaMalloc(ptrB, p * n * Sizeof.DOUBLE.toLong())
        cudaMalloc(ptrC, m * n * Sizeof.DOUBLE.toLong())

        cublasSetVector(m * p, Sizeof.DOUBLE, Pointer.to(A), 1, ptrA, 1)
        cublasSetVector(p * n, Sizeof.DOUBLE, Pointer.to(B), 1, ptrB, 1)
        cublasSetVector(m * n, Sizeof.DOUBLE, Pointer.to(C), 1, ptrC, 1)

        JCublas2.cublasDgemm(handle, cublasOperation.CUBLAS_OP_N,
            cublasOperation.CUBLAS_OP_N, m, n, p, ptrAlpha, ptrA, CUDA_R_64F,
            ptrB, CUDA_R_64F, ptrBeta, ptrC, CUDA_R_64F)

        cudaDeviceSynchronize()

        cublasGetVector(m * n, Sizeof.FLOAT, ptrC, 1, Pointer.to(C), 1)

        cudaFree(ptrA)
        cudaFree(ptrB)
        cudaFree(ptrC)
        cublasDestroy(handle)

        var index = 0
        return Vector.ofVectors(m) {
            Vector(n) { C[index++] }
        }
    }
}
