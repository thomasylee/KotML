package kotml.math

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class MatrixTest {
    @Test
    fun `Matrix(vararg Double) returns Matrix with correct values`() {
        val matrix = Matrix(-1.5, 0.0, 1.5)
        assertEquals(1, matrix.dimensions)
        assertEquals(1, matrix.shape.size)
        assertTrue(matrix.shapeEquals(intArrayOf(3)))
        assertEquals(-1.5, matrix(0))
        assertEquals(0.0, matrix(1))
        assertEquals(1.5, matrix(2))
    }

    @Test
    fun `Matrix(vararg Matrix) returns Matrix with correct values`() {
        val matrix = Matrix(
            Matrix(-2.0, -1.0),
            Matrix(0.0, 1.0),
            Matrix(2.0, 3.0))
        assertEquals(2, matrix.dimensions)
        assertEquals(2, matrix.shape.size)
        assertTrue(matrix.shapeEquals(intArrayOf(3, 2)))
        assertEquals(-2.0, matrix[0](0))
        assertEquals(-1.0, matrix[0](1))
        assertEquals(0.0, matrix[1](0))
        assertEquals(1.0, matrix[1](1))
        assertEquals(2.0, matrix[2](0))
        assertEquals(3.0, matrix[2](1))
    }

    @Test
    fun `Matrix(Matrix) initializes as a copy of the matrix`() {
        assertEquals(Matrix(1.0, 2.0, 3.0), Matrix(Matrix(1.0, 2.0, 3.0)))
        assertEquals(Matrix(Matrix(1.0), Matrix(2.0)),
            Matrix(Matrix(Matrix(1.0), Matrix(2.0))))
    }

    @Test
    fun `transpose() correctly transposes the matrix`() {
        assertEquals(Matrix(Matrix(1.0), Matrix(2.0)), Matrix(1.0, 2.0).transpose())
        assertEquals(Matrix(1.0, 2.0), Matrix(Matrix(1.0), Matrix(2.0)).transpose())
        assertEquals(
            Matrix(
                Matrix(1.0, 4.0),
                Matrix(2.0, 5.0),
                Matrix(3.0, 6.0)),
            Matrix(
                Matrix(1.0, 2.0, 3.0),
                Matrix(4.0, 5.0, 6.0)).transpose())
    }

    @Test
    fun `det() returns the determinant for square matrices`() {
        assertEquals(-2.0, Matrix(Matrix(1.0, 2.0), Matrix(3.0, 4.0)).det())
        assertEquals(12.0, Matrix(
            Matrix(-1.0, 1.0, 3.0),
            Matrix(2.0, 1.0, 0.0),
            Matrix(4.0, 3.0, -2.0)).det())
    }

    @Test
    fun `clone() creates a copy of the matrix`() {
        val rowMatrix = Matrix(1.0, 2.0, 3.0)
        val rowMatrixCopy = rowMatrix.clone()
        assertEquals(Matrix(1.0, 2.0, 3.0), rowMatrixCopy)

        val multidimensionalMatrix = Matrix(
            Matrix(1.0, 2.0),
            Matrix(3.0, 4.0))
        val multidimensionalMatrixCopy = multidimensionalMatrix.clone()
        assertEquals(Matrix(Matrix(1.0, 2.0), Matrix(3.0, 4.0)), multidimensionalMatrixCopy)
    }

    @Test
    fun `== returns correct result`() {
        val matrix1 = Matrix(
            Matrix(-2.0, -1.0),
            Matrix(0.0, 1.0),
            Matrix(2.0, 3.0))
        val matrix2 = Matrix(
            Matrix(-2.0, -1.0),
            Matrix(0.0, 1.0),
            Matrix(2.0, 3.0))
        val matrix3 = Matrix(1.0, 2.0, 3.0)

        assertTrue(matrix1 == matrix2)
        assertFalse(matrix1 != matrix2)

        assertTrue(matrix1 != matrix3)
        assertFalse(matrix1 == matrix3)
    }

    @Test
    fun `unaryMinus() returns correct result`() {
        val matrix1 = Matrix(
            Matrix(-2.0, -1.0),
            Matrix(0.0, 1.0),
            Matrix(-4.0, 3.0))
        val matrix2 = Matrix(
            Matrix(2.0, 1.0),
            Matrix(0.0, -1.0),
            Matrix(4.0, -3.0))

        assertEquals(matrix2, -matrix1)
    }

    @Test
    fun `+ adds matrices correctly`() {
        val matrix1 = Matrix(
            Matrix(-1.0, 0.0, 1.0),
            Matrix(1.0, 0.0, -1.0))
        val matrix2 = Matrix(
            Matrix(3.0, 4.0, 5.0),
            Matrix(0.0, 1.0, 2.0))
        val matrix3 = Matrix(
            Matrix(2.0, 4.0, 6.0),
            Matrix(1.0, 1.0, 1.0))

        assertEquals(matrix3, matrix1 + matrix2)
    }

    @Test
    fun `- subtracts matrices correctly`() {
        val matrix1 = Matrix(
            Matrix(-1.0, 0.0, 1.0),
            Matrix(1.0, 0.0, -1.0))
        val matrix2 = Matrix(
            Matrix(3.0, 4.0, 5.0),
            Matrix(0.0, 1.0, 2.0))
        val matrix3 = Matrix(
            Matrix(-4.0, -4.0, -4.0),
            Matrix(1.0, -1.0, -3.0))

        assertEquals(matrix3, matrix1 - matrix2)
    }

    @Test
    fun `* multiplies matrices correctly`() {
        val matrix1 = Matrix(
            Matrix(-1.0, 0.0, 2.0),
            Matrix(1.0, 0.0, -1.0))
        val matrix2 = Matrix(
            Matrix(3.0, 4.0, 5.0),
            Matrix(0.0, 1.0, 2.0))
        val matrix3 = Matrix(
            Matrix(-3.0, 0.0, 10.0),
            Matrix(0.0, 0.0, -2.0))

        assertEquals(matrix3, matrix1 * matrix2)
    }

    @Test
    fun `div() divides matrices correctly`() {
        val matrix1 = Matrix(
            Matrix(-4.0, 9.0, 1.0),
            Matrix(0.0, -15.0, 5.0))
        val matrix2 = Matrix(
            Matrix(-2.0, 3.0, 1.0),
            Matrix(2.0, 5.0, -1.0))
        val matrix3 = Matrix(
            Matrix(2.0, 3.0, 1.0),
            Matrix(0.0, -3.0, -5.0))

        assertEquals(matrix3, matrix1 / matrix2)
    }

    @Test
    fun `x returns matrix multiplication product for 1x1 matrix`() {
        val matrix1 = Matrix(-2.0)
        val matrix2 = Matrix(-3.0)
        val matrix3 = Matrix(6.0)

        assertEquals(matrix3, matrix1 x matrix2)
    }

    @Test
    fun `x returns matrix multiplication product for row matrix`() {
        val matrix1 = Matrix(-4.0, 9.0, 1.0)
        val matrix2 = Matrix(
            Matrix(-2.0),
            Matrix(1.0),
            Matrix(2.0))
        val matrix3 = Matrix(19.0)

        assertEquals(matrix3, matrix1 x matrix2)
    }

    @Test
    fun `x returns matrix multiplication product for multidimensional matrices`() {
        val matrix1 = Matrix(
            Matrix(-4.0, 9.0, 1.0),
            Matrix(0.0, -5.0, 5.0),
            Matrix(1.0, -1.0, 2.0))
        val matrix2 = Matrix(
            Matrix(-2.0, 3.0),
            Matrix(1.0, 0.0),
            Matrix(2.0, 1.0))
        val matrix3 = Matrix(
            Matrix(19.0, -11.0),
            Matrix(5.0, 5.0),
            Matrix(1.0, 5.0))

        assertEquals(matrix3, matrix1 x matrix2)
    }

    @Test
    fun `x throws ShapeException for invalid matrix dimensions`() {
        assertThrows(ShapeException::class.java) {
            Matrix(1.0) x Matrix(1.0, 2.0)
        }

        assertThrows(ShapeException::class.java) {
            Matrix(1.0, 2.0) x Matrix(1.0, 2.0)
        }

        assertThrows(ShapeException::class.java) {
            Matrix(Matrix(1.0), Matrix(2.0)) x Matrix(1.0, 2.0)
        }

        assertThrows(ShapeException::class.java) {
            Matrix(Matrix(1.0), Matrix(2.0)) x Matrix(Matrix(1.0), Matrix(2.0))
        }

        assertThrows(ShapeException::class.java) {
            Matrix(
                Matrix(1.0, 2.0, 3.0),
                Matrix(1.0, 2.0, 3.0)) x Matrix(
                    Matrix(1.0, 2.0),
                    Matrix(1.0, 2.0))
        }
    }
}
