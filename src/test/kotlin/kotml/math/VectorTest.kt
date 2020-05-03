package kotml.math

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class VectorTest {
    @Test
    fun `Vector(vararg Double) returns Vector with correct values`() {
        val vector = Vector(-1.5, 0.0, 1.5)
        assertEquals(1, vector.dimensions)
        assertEquals(1, vector.shape.size)
        assertTrue(vector.shapeEquals(intArrayOf(3)))
        assertEquals(-1.5, vector(0))
        assertEquals(0.0, vector(1))
        assertEquals(1.5, vector(2))
    }

    @Test
    fun `Vector(vararg Vector) returns Vector with correct values`() {
        val vector = Vector(
            Vector(-2.0, -1.0),
            Vector(0.0, 1.0),
            Vector(2.0, 3.0))
        assertEquals(2, vector.dimensions)
        assertEquals(2, vector.shape.size)
        assertTrue(vector.shapeEquals(intArrayOf(3, 2)))
        assertEquals(-2.0, vector[0](0))
        assertEquals(-1.0, vector[0](1))
        assertEquals(0.0, vector[1](0))
        assertEquals(1.0, vector[1](1))
        assertEquals(2.0, vector[2](0))
        assertEquals(3.0, vector[2](1))
    }

    @Test
    fun `Vector(Vector) initializes as a copy of the vector`() {
        assertEquals(Vector(1.0, 2.0, 3.0), Vector(Vector(1.0, 2.0, 3.0)))
        assertEquals(Vector(Vector(1.0), Vector(2.0)),
            Vector(Vector(Vector(1.0), Vector(2.0))))
    }

    @Test
    fun `transpose() correctly transposes the vector`() {
        assertEquals(Vector(Vector(1.0), Vector(2.0)), Vector(1.0, 2.0).transpose())
        assertEquals(Vector(1.0, 2.0), Vector(Vector(1.0), Vector(2.0)).transpose())
        assertEquals(
            Vector(
                Vector(1.0, 4.0),
                Vector(2.0, 5.0),
                Vector(3.0, 6.0)),
            Vector(
                Vector(1.0, 2.0, 3.0),
                Vector(4.0, 5.0, 6.0)).transpose())
    }

    @Test
    fun `det() returns the determinant for square matrices`() {
        assertEquals(-2.0, Vector(Vector(1.0, 2.0), Vector(3.0, 4.0)).det())
        assertEquals(12.0, Vector(
            Vector(-1.0, 1.0, 3.0),
            Vector(2.0, 1.0, 0.0),
            Vector(4.0, 3.0, -2.0)).det())
    }

    @Test
    fun `clone() creates a copy of the vector`() {
        val rowVector = Vector(1.0, 2.0, 3.0)
        val rowVectorCopy = rowVector.clone()
        assertEquals(Vector(1.0, 2.0, 3.0), rowVectorCopy)

        val multidimensionalVector = Vector(
            Vector(1.0, 2.0),
            Vector(3.0, 4.0))
        val multidimensionalVectorCopy = multidimensionalVector.clone()
        assertEquals(Vector(Vector(1.0, 2.0), Vector(3.0, 4.0)), multidimensionalVectorCopy)
    }

    @Test
    fun `== returns correct result`() {
        val vector1 = Vector(
            Vector(-2.0, -1.0),
            Vector(0.0, 1.0),
            Vector(2.0, 3.0))
        val vector2 = Vector(
            Vector(-2.0, -1.0),
            Vector(0.0, 1.0),
            Vector(2.0, 3.0))
        val vector3 = Vector(1.0, 2.0, 3.0)

        assertTrue(vector1 == vector2)
        assertFalse(vector1 != vector2)

        assertTrue(vector1 != vector3)
        assertFalse(vector1 == vector3)
    }

    @Test
    fun `unaryMinus() returns correct result`() {
        val vector1 = Vector(
            Vector(-2.0, -1.0),
            Vector(0.0, 1.0),
            Vector(-4.0, 3.0))
        val vector2 = Vector(
            Vector(2.0, 1.0),
            Vector(0.0, -1.0),
            Vector(4.0, -3.0))

        assertEquals(vector2, -vector1)
    }

    @Test
    fun `+ adds vectors correctly`() {
        val vector1 = Vector(
            Vector(-1.0, 0.0, 1.0),
            Vector(1.0, 0.0, -1.0))
        val vector2 = Vector(
            Vector(3.0, 4.0, 5.0),
            Vector(0.0, 1.0, 2.0))
        val vector3 = Vector(
            Vector(2.0, 4.0, 6.0),
            Vector(1.0, 1.0, 1.0))

        assertEquals(vector3, vector1 + vector2)
    }

    @Test
    fun `- subtracts vectors correctly`() {
        val vector1 = Vector(
            Vector(-1.0, 0.0, 1.0),
            Vector(1.0, 0.0, -1.0))
        val vector2 = Vector(
            Vector(3.0, 4.0, 5.0),
            Vector(0.0, 1.0, 2.0))
        val vector3 = Vector(
            Vector(-4.0, -4.0, -4.0),
            Vector(1.0, -1.0, -3.0))

        assertEquals(vector3, vector1 - vector2)
    }

    @Test
    fun `* multiplies vectors correctly`() {
        val vector1 = Vector(
            Vector(-1.0, 0.0, 2.0),
            Vector(1.0, 0.0, -1.0))
        val vector2 = Vector(
            Vector(3.0, 4.0, 5.0),
            Vector(0.0, 1.0, 2.0))
        val vector3 = Vector(
            Vector(-3.0, 0.0, 10.0),
            Vector(0.0, 0.0, -2.0))

        assertEquals(vector3, vector1 * vector2)
    }

    @Test
    fun `div() divides vectors correctly`() {
        val vector1 = Vector(
            Vector(-4.0, 9.0, 1.0),
            Vector(0.0, -15.0, 5.0))
        val vector2 = Vector(
            Vector(-2.0, 3.0, 1.0),
            Vector(2.0, 5.0, -1.0))
        val vector3 = Vector(
            Vector(2.0, 3.0, 1.0),
            Vector(0.0, -3.0, -5.0))

        assertEquals(vector3, vector1 / vector2)
    }

    @Test
    fun `x returns matrix multiplication product for 1x1 vector`() {
        val vector1 = Vector(-2.0)
        val vector2 = Vector(-3.0)
        val vector3 = Vector(6.0)

        assertEquals(vector3, vector1 x vector2)
    }

    @Test
    fun `x returns matrix multiplication product for row vector`() {
        val vector1 = Vector(-4.0, 9.0, 1.0)
        val vector2 = Vector(
            Vector(-2.0),
            Vector(1.0),
            Vector(2.0))
        val vector3 = Vector(19.0)

        assertEquals(vector3, vector1 x vector2)
    }

    @Test
    fun `x returns matrix multiplication product for multidimensional vectors`() {
        val vector1 = Vector(
            Vector(-4.0, 9.0, 1.0),
            Vector(0.0, -5.0, 5.0),
            Vector(1.0, -1.0, 2.0))
        val vector2 = Vector(
            Vector(-2.0, 3.0),
            Vector(1.0, 0.0),
            Vector(2.0, 1.0))
        val vector3 = Vector(
            Vector(19.0, -11.0),
            Vector(5.0, 5.0),
            Vector(1.0, 5.0))

        assertEquals(vector3, vector1 x vector2)
    }

    @Test
    fun `x throws ShapeException for invalid vector dimensions`() {
        assertThrows(ShapeException::class.java) {
            Vector(1.0) x Vector(1.0, 2.0)
        }

        assertThrows(ShapeException::class.java) {
            Vector(1.0, 2.0) x Vector(1.0, 2.0)
        }

        assertThrows(ShapeException::class.java) {
            Vector(Vector(1.0), Vector(2.0)) x Vector(1.0, 2.0)
        }

        assertThrows(ShapeException::class.java) {
            Vector(Vector(1.0), Vector(2.0)) x Vector(Vector(1.0), Vector(2.0))
        }

        assertThrows(ShapeException::class.java) {
            Vector(
                Vector(1.0, 2.0, 3.0),
                Vector(1.0, 2.0, 3.0)) x Vector(
                    Vector(1.0, 2.0),
                    Vector(1.0, 2.0))
        }
    }

    @Test
    fun `sum(axis) returns the sum along the correct axis`() {
        assertEquals(Vector(2.0), Vector(2.0).sum(0))
        assertEquals(Vector(3.0), Vector(1.0, 2.0).sum(0))
        assertEquals(
            Vector(5.0, 7.0, 9.0),
            Vector(Vector(1.0, 2.0, 3.0), Vector(4.0, 5.0, 6.0)).sum(0))
        assertEquals(
            Vector(6.0, 15.0),
            Vector(Vector(1.0, 2.0, 3.0), Vector(4.0, 5.0, 6.0)).sum(1))
    }

    @Test
    fun `product(axis) returns the product along the correct axis`() {
        assertEquals(Vector(2.0), Vector(2.0).product(0))
        assertEquals(Vector(-2.0), Vector(-1.0, 2.0).product(0))
        assertEquals(
            Vector(4.0, 10.0, 18.0),
            Vector(Vector(1.0, 2.0, 3.0), Vector(4.0, 5.0, 6.0)).product(0))
        assertEquals(
            Vector(6.0, -120.0),
            Vector(Vector(1.0, 2.0, 3.0), Vector(-4.0, 5.0, 6.0)).product(1))
    }

    @Test
    fun `dot() returns the correct dot product`() {
        assertEquals(5.0, Vector(1.0, 2.0, 3.0) dot Vector(-1.0, 0.0, 2.0))
    }

    @Test
    fun `transpose() returns the transpose of the vector`() {
        assertEquals(Vector(Vector(1.0), Vector(2.0)), Vector(1.0, 2.0).transpose())
        assertEquals(Vector(1.0, 2.0), Vector(Vector(1.0), Vector(2.0)).transpose())
        assertEquals(
            Vector(Vector(1.0, 4.0), Vector(2.0, 5.0), Vector(3.0, 6.0)),
            Vector(Vector(1.0, 2.0, 3.0), Vector(4.0, 5.0, 6.0)).transpose())
    }
}
