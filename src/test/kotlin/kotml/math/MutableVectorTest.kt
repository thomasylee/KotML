package kotml.math

import kotlin.test.assertFailsWith
import kotml.extensions.div
import kotml.extensions.minus
import kotml.extensions.plus
import kotml.extensions.times
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotEquals
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
    fun `ofVectors() initializes a MutableVector of MutableVectors`() {
        val vector = MutableVector.ofVectors(2) { MutableVector(it, it) }
        assertTrue(vector(0) is MutableVector, "Subvector should be a MutableVector")
        assertEquals(Vector(Vector(0, 0), Vector(1, 1)), vector)
    }

    @Test
    fun `copy() returns a new MutableVector`() {
        val original = MutableVector(Vector(1, 2), Vector(3, 4))
        val copy = original.copy()
        original[0, 0] = 5
        assertEquals(Vector(Vector(5, 2), Vector(3, 4)), original)
        assertEquals(Vector(Vector(1, 2), Vector(3, 4)), copy)
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

    @Test
    fun `plusAssign() adds to all elements`() {
        val vector = MutableVector(Vector(1, 2), Vector(3, 4))
        vector += 1
        assertEquals(Vector(Vector(2, 3), Vector(4, 5)), vector)
        vector += 2.0
        assertEquals(Vector(Vector(4, 5), Vector(6, 7)), vector)
    }

    @Test
    fun `minusAssign() subtracts from all elements`() {
        val vector = MutableVector(Vector(4, 5), Vector(6, 7))
        vector -= 1
        assertEquals(Vector(Vector(3, 4), Vector(5, 6)), vector)
        vector -= 2.0
        assertEquals(Vector(Vector(1, 2), Vector(3, 4)), vector)
    }

    @Test
    fun `timesAssign() multiples to all elements`() {
        val vector = MutableVector(Vector(4, 5), Vector(6, 7))
        vector *= 2
        assertEquals(Vector(Vector(8, 10), Vector(12, 14)), vector)
        vector *= 3.0
        assertEquals(Vector(Vector(24, 30), Vector(36, 42)), vector)
    }

    @Test
    fun `divAssign() divides from all elements`() {
        val vector = MutableVector(Vector(24, 30), Vector(36, 42))
        vector /= 2
        assertEquals(Vector(Vector(12, 15), Vector(18, 21)), vector)
        vector /= 3.0
        assertEquals(Vector(Vector(4, 5), Vector(6, 7)), vector)
    }

    @Test
    fun `toMutableVector() returns this instead of creating a new vector`() {
        val vector = Vector(1, 2, 3)
        val mutable = vector.toMutableVector()
        assertNotEquals(vector.hashCode(), mutable.hashCode())
        assertEquals(mutable.hashCode(), mutable.toMutableVector().hashCode())
    }
}
