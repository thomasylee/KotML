package kotml.math

import kotlin.test.assertFailsWith
import kotml.extensions.div
import kotml.extensions.minus
import kotml.extensions.plus
import kotml.extensions.times
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class MutableVectorTest {
    @Test
    fun `MutableVector(shape, mapValues) initializes with correct values`() {
        val vector = MutableVector(2, 3) { it * 2.0 }
        assertEquals(2, vector.dimensions)
        assertEquals(2, vector.shape.size)
        assertTrue(vector.shapeEquals(intArrayOf(2, 3)))
        assertEquals(0.0, vector[0, 0])
        assertEquals(2.0, vector(0)[1])
        assertEquals(4.0, vector[0, 2])
        assertEquals(6.0, vector(1)[0])
        assertEquals(8.0, vector[1, 1])
        assertEquals(10.0, vector(1)[2])
    }

    @Test
    fun `MutableVector(vararg Double) initializes with correct values`() {
        val vector = MutableVector(-1.5, 0.0, 1.5)
        assertEquals(1, vector.dimensions)
        assertEquals(1, vector.shape.size)
        assertTrue(vector.shapeEquals(intArrayOf(3)))
        assertEquals(-1.5, vector[0])
        assertEquals(0.0, vector[1])
        assertEquals(1.5, vector[2])
    }

    @Test
    fun `MutableVector(vararg Int) initializes with correct values`() {
        val vector = MutableVector(-1, 0, 1)
        assertEquals(1, vector.dimensions)
        assertEquals(1, vector.shape.size)
        assertTrue(vector.shapeEquals(intArrayOf(3)))
        assertEquals(-1.0, vector[0])
        assertEquals(0.0, vector[1])
        assertEquals(1.0, vector[2])
    }

    @Test
    fun `MutableVector(vararg Vector) initializes with correct values`() {
        val vector = MutableVector(
            Vector(-2, -1),
            Vector(0.0, 1.0),
            Vector(2.0, 3.0))
        assertEquals(2, vector.dimensions)
        assertEquals(2, vector.shape.size)
        assertTrue(vector.shapeEquals(intArrayOf(3, 2)))
        assertEquals(-2.0, vector[0, 0])
        assertEquals(-1.0, vector[0, 1])
        assertEquals(0.0, vector[1, 0])
        assertEquals(1.0, vector[1, 1])
        assertEquals(2.0, vector(2)[0])
        assertEquals(3.0, vector(2)[1])
    }

    @Test
    fun `zeros() initializes a MutableVector with all zero values`() {
        assertEquals(MutableVector(0.0, 0.0, 0.0), MutableVector.zeros(3))
        assertEquals(
            MutableVector(MutableVector(0, 0), MutableVector(0, 0), MutableVector(0, 0)),
            MutableVector.zeros(3, 2))
    }

    @Test
    fun `set() sets scalar values correctly`() {
        val rowVector = MutableVector(1, 2, 3)
        rowVector[1] = 5.0
        assertEquals(Vector(1, 5, 3), rowVector)

        val squareVector = MutableVector(Vector(1, 2), Vector(3, 4))
        squareVector[0, 1] = 5.0
        assertEquals(Vector(Vector(1, 5), Vector(3, 4)), squareVector)
    }

    @Test
    fun `set() sets vector values correctly`() {
        val squareVector = MutableVector(Vector(1, 2), Vector(3, 4))
        squareVector[1] = Vector(5, 6)
        assertEquals(Vector(Vector(1, 2), Vector(5, 6)), squareVector)
    }

    @Test
    fun `set() raises ShapeException for invalid shapes`() {
        val vector = MutableVector(Vector(1, 2, 3), Vector(4, 5, 6))

        assertFailsWith(ShapeException::class) {
            vector[0] = 5.0
        }
        assertFailsWith(ShapeException::class) {
            vector[0, 0, 0] = 5.0
        }
        assertFailsWith(ShapeException::class) {
            vector[0, 0] = Vector(1, 2, 3)
        }
        assertFailsWith(ShapeException::class) {
            vector[0] = Vector(1, 2)
        }
        assertFailsWith(ShapeException::class) {
            vector[0] = Vector(1, 2, 3, 4)
        }
    }
}
