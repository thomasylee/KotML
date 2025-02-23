package kotml.math

import kotlin.random.Random
import kotml.extensions.div
import kotml.extensions.minus
import kotml.extensions.plus
import kotml.extensions.times
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class VectorTest {
    @Test
    fun `Vector(shape, mapValues) initializes Vector with correct values`() {
        val vector = Vector(2, 3, 4) { it * 2.0 }
        assertEquals(3, vector.dimensions)
        assertEquals(3, vector.shape.size)
        assertTrue(vector.shapeEquals(intArrayOf(2, 3, 4)))
        assertEquals(0.0, vector[0, 0, 0])
        assertEquals(2.0, vector(0)(0)[1])
        assertEquals(8.0, vector(0, 1)[0])
        assertEquals(10.0, vector[0, 1, 1])
        assertEquals(24.0, vector[1, 0, 0])
        assertEquals(26.0, vector[1, 0, 1])
        assertEquals(32.0, vector[1, 1, 0])
        assertEquals(34.0, vector[1, 1, 1])
        assertEquals(46.0, vector[1, 2, 3])
    }

    @Test
    fun `Vector(vararg Double) initializes Vector with correct values`() {
        val vector = Vector(-1.5, 0.0, 1.5)
        assertEquals(1, vector.dimensions)
        assertEquals(1, vector.shape.size)
        assertTrue(vector.shapeEquals(intArrayOf(3)))
        assertEquals(-1.5, vector[0])
        assertEquals(0.0, vector[1])
        assertEquals(1.5, vector[2])
    }

    @Test
    fun `Vector(vararg Int) initializes Vector with correct values`() {
        val vector = Vector(-1, 0, 1)
        assertEquals(1, vector.dimensions)
        assertEquals(1, vector.shape.size)
        assertTrue(vector.shapeEquals(intArrayOf(3)))
        assertEquals(-1.0, vector[0])
        assertEquals(0.0, vector[1])
        assertEquals(1.0, vector[2])
    }

    @Test
    fun `Vector(vararg Vector) initializes Vector with correct values`() {
        val vector = Vector(
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
    fun `retrieving child vectors by one dimension works`() {
        assertEquals(
            Vector(1, 2, 3),
            Vector(Vector(1, 2, 3), Vector(4, 5, 6))(0))
        assertEquals(
            Vector(3, 4),
            Vector(
                Vector(Vector(1, 2), Vector(3, 4)),
                Vector(5, 6), Vector(7, 8))(0, 1))
    }

    @Test
    fun `retrieving child vectors by multiple dimensions works`() {
        assertEquals(
            Vector(
                Vector(4, 5, 6),
                Vector(13, 14, 15)),
            Vector(
                Vector(Vector(1, 2, 3), Vector(4, 5, 6), Vector(7, 8, 9)),
                Vector(Vector(10, 11, 12), Vector(13, 14, 15), Vector(16, 17, 18)))(null, 1, null))
    }

    @Test
    fun `retrieving scalar values works`() {
        assertEquals(5.0, Vector(5)[0])
        assertEquals(2.0, Vector(1, 2, 3)[1])
    }

    @Test
    fun `zeros() initializes a vector with all zero values`() {
        assertEquals(Vector(0.0, 0.0, 0.0), Vector.zeros(3))
        assertEquals(
            Vector(Vector(0, 0), Vector(0, 0), Vector(0, 0)),
            Vector.zeros(3, 2))
    }

    @Test
    fun `random() initializes a vector with random values`() {
        val random = Random(0)
        assertEquals(
            Vector(0.54963315022148, 0.8488554755054833, 0.6591029791044284),
            Vector.random(3, random = random))
        assertEquals(
            Vector(
                Vector(0.0079879341867376, 0.3249690959322964),
                Vector(0.010111317974326073, 0.5345753913241595),
                Vector(0.6916786784731803, 0.948378936177206)),
            Vector.random(3, 2, random = random))
    }

    @Test
    fun `ofVectors() initializes a vector of vectors`() {
        assertEquals(Vector(Vector(0, 0), Vector(1, 1)), Vector.ofVectors(2) {
            Vector(it, it)
        })
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
    fun `inverse() returns the inverse of the square matrix`() {
        assertEquals(Vector(0.5), Vector(2).inverse())
        assertEquals(
            Vector(Vector(0.6, -0.7), Vector(-0.2, 0.4)),
            Vector(Vector(4, 7), Vector(2, 6)).inverse())
        assertEquals(
            Vector(
                Vector(0.2, 0.2, 0.0),
                Vector(-0.2, 0.3, 1.0),
                Vector(0.2, -0.3, 0.0)),
            Vector(
                Vector(3, 0, 2),
                Vector(2, 0, -2),
                Vector(0, 1, 1)).inverse())
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
    fun `+ adds doubles and vectors correctly`() {
        assertEquals(Vector(2.0, 3.0), 1.0 + Vector(1.0, 2.0))
        assertEquals(Vector(Vector(2.0), Vector(3.0)),
            Vector(Vector(1.0), Vector(2.0)) + 1.0)
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
    fun `- subtracts doubles and vectors correctly`() {
        assertEquals(Vector(0.0, -1.0), 1.0 - Vector(1.0, 2.0))
        assertEquals(Vector(Vector(0.0), Vector(1.0)),
            Vector(Vector(1.0), Vector(2.0)) - 1.0)
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
    fun `* multiplies doubles and vectors correctly`() {
        assertEquals(Vector(1.0, 2.0), 1.0 * Vector(1.0, 2.0))
        assertEquals(Vector(Vector(1.0), Vector(2.0)),
            Vector(Vector(1.0), Vector(2.0)) * 1.0)
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
    fun `div() divides doubles and vectors correctly`() {
        assertEquals(Vector(1.0, 0.5), 1.0 / Vector(1.0, 2.0))
        assertEquals(Vector(Vector(0.5), Vector(1.0)),
            Vector(Vector(1.0), Vector(2.0)) / 2.0)
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
    fun `forEachIndexed() loops through all elements with indices`() {
        val vector = Vector(Vector(1, 2), Vector(3, 4))
        var realIndex = 0
        vector.forEachIndexed { index, value ->
            assertEquals(realIndex, index)
            assertEquals(realIndex + 1.0, value)
            realIndex++
        }
        // Make sure we looped through all the elements
        assertEquals(vector.shape[0] * vector.shape[1], realIndex)
    }

    @Test
    fun `forEach() loops through all elements`() {
        val vector = Vector(Vector(1, 2), Vector(3, 4))
        var realIndex = 0
        vector.forEach { value ->
            assertEquals(realIndex + 1.0, value)
            realIndex++
        }
        // Make sure we looped through all the elements
        assertEquals(vector.shape[0] * vector.shape[1], realIndex)
    }

    @Test
    fun `mapIndexed() maps with index correctly`() {
        assertEquals(
            Vector(Vector(0.0, 2.0), Vector(4.0, 6.0)),
            Vector.zeros(2, 2).mapIndexed { index, value ->
                value + index * 2.0
            }
        )
    }

    @Test
    fun `map() maps correctly`() {
        assertEquals(
            Vector(Vector(1.0, 1.0), Vector(1.0, 1.0)),
            Vector.zeros(2, 2).map { value ->
                value + 1.0
            }
        )
    }

    @Test
    fun `foldIndexed() folds with correct indices`() {
        assertEquals(
            Vector(25),
            // 5 + (0 * 1 + 1 * 2 + 2 * 3 + 3 * 4) = (25)
            Vector(1, 2, 3, 4).foldIndexed(5.0) { index, acc, value ->
                acc + index * value
            }
        )
        assertEquals(
            Vector(17, 27, 41),
            // 5 + ([0 * 1 + 3 * 4], [1 * 2 + 4 * 5], [2 * 3 + 5 * 6]) =
            // (17, 27, 41)
            Vector(Vector(1, 2, 3), Vector(4, 5, 6)).foldIndexed(5.0, axis = 0) { index, acc, value ->
                acc + index * value
            }
        )
        assertEquals(
            Vector(13, 67),
            // 5 + ([0 * 1 + 1 * 2 + 2 * 3], [3 * 4 + 4 * 5 + 5 * 6]) =
            // (13, 67)
            Vector(Vector(1, 2, 3), Vector(4, 5, 6)).foldIndexed(5.0, axis = 1) { index, acc, value ->
                acc + index * value
            }
        )
    }

    @Test
    fun `fold() folds correctly`() {
        assertEquals(
            Vector(15),
            Vector(1, 2, 3, 4).fold(5.0) { acc, value -> acc + value }
        )
        assertEquals(
            Vector(4, 10, 18),
            Vector(Vector(1, 2, 3), Vector(4, 5, 6)).fold(1.0, axis = 0) { acc, value ->
                acc * value
            }
        )
        assertEquals(
            Vector(6, 120),
            Vector(Vector(1, 2, 3), Vector(4, 5, 6)).fold(1.0, axis = 1) { acc, value ->
                acc * value
            }
        )
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
    fun `max(axis) returns the maximum along the correct axis`() {
        assertEquals(Vector(2.0), Vector(2.0).max(0))
        assertEquals(Vector(2.0), Vector(-1.0, 2.0).max(0))
        assertEquals(
            Vector(1.0, 6.0, 7.0),
            Vector(Vector(1.0, 2.0, 7.0), Vector(1.0, 6.0, 5.0)).max(0))
        assertEquals(
            Vector(7.0, 6.0),
            Vector(Vector(1.0, 2.0, 7.0), Vector(1.0, 6.0, 5.0)).max(1))
    }

    @Test
    fun `min(axis) returns the minimum along the correct axis`() {
        assertEquals(Vector(2.0), Vector(2.0).min(0))
        assertEquals(Vector(-1.0), Vector(-1.0, 2.0).min(0))
        assertEquals(
            Vector(1.0, 2.0, 5.0),
            Vector(Vector(1.0, 2.0, 7.0), Vector(1.0, 6.0, 5.0)).min(0))
        assertEquals(
            Vector(1.0, 1.0),
            Vector(Vector(1.0, 2.0, 7.0), Vector(1.0, 6.0, 5.0)).min(1))
    }

    @Test
    fun `argmax() returns the index with the largest value`() {
        assertEquals(3, Vector(-2, 7, 3, 8, 1).argmax())

        // Ensure random selection to break ties.
        val largeVector = Vector.zeros(10, 10)
        val try1 = largeVector.argmax()
        val try2 = largeVector.argmax()
        val try3 = largeVector.argmax()
        val try4 = largeVector.argmax()
        assertFalse(try1 == try2 && try2 == try3 && try3 == try4)
    }

    @Test
    fun `dot() returns the correct dot product`() {
        assertEquals(5.0, Vector(1.0, 2.0, 3.0) dot Vector(-1.0, 0.0, 2.0))
    }

    @Test
    fun `transpose() returns the transpose of the vector`() {
        assertEquals(Vector(5.0), Vector(5.0).transpose())
        assertEquals(Vector(Vector(1.0), Vector(2.0)), Vector(1.0, 2.0).transpose())
        assertEquals(Vector(1.0, 2.0), Vector(Vector(1.0), Vector(2.0)).transpose())
        assertEquals(
            Vector(Vector(1.0, 4.0), Vector(2.0, 5.0), Vector(3.0, 6.0)),
            Vector(Vector(1.0, 2.0, 3.0), Vector(4.0, 5.0, 6.0)).transpose())
    }

    @Test
    fun `flatten() returns a row vector with the same scalar values`() {
        assertEquals(Vector(1, 2, 3, 4, 5, 6, 7, 8), Vector(
            Vector(Vector(1, 2), Vector(3, 4)),
            Vector(Vector(5, 6), Vector(7, 8))).flatten())
    }

    @Test
    fun `reshape() returns a vector with the desired shape`() {
        assertEquals(
            Vector(1, 2, 3, 4),
            Vector(Vector(1, 2), Vector(3, 4)).reshape(4))
        assertEquals(
            Vector(Vector(Vector(1), Vector(2), Vector(3), Vector(4))),
            Vector(Vector(1, 2), Vector(3, 4)).reshape(1, 4, 1))
        assertEquals(
            Vector(Vector(1, 2), Vector(3, 4)),
            Vector(1, 2, 3, 4).reshape(2, 2))
    }

    @Test
    fun `reshape() throws a ShapeException if the shapes are incompatible`() {
        assertThrows(ShapeException::class.java) {
            Vector(1, 2, 3).reshape(2, 2)
        }
    }

    @Test
    fun `shuffle() with axis = null returns a completely shuffled vector`() {
        val random = Random(0)
        assertEquals(Vector(4, 2, 1, 3), Vector(1, 2, 3, 4).shuffle(random = random))
        assertEquals(
            Vector(
                Vector(Vector(9, 11, 10), Vector(1, 12, 6)),
                Vector(Vector(3, 2, 7), Vector(8, 5, 4))),
            Vector(
                Vector(Vector(1, 2, 3), Vector(4, 5, 6)),
                Vector(Vector(7, 8, 9), Vector(10, 11, 12))).shuffle(random = random))
    }

    @Test
    fun `shuffle() with non-null axis returns a vector shuffled on the right axis`() {
        val random = Random(0)
        assertEquals(Vector(4, 2, 1, 3), Vector(1, 2, 3, 4).shuffle(0, random))
        assertEquals(
            Vector(
                Vector(Vector(19, 20, 21), Vector(22, 23, 24), Vector(25, 26, 27)),
                Vector(Vector(10, 11, 12), Vector(13, 14, 15), Vector(16, 17, 18)),
                Vector(Vector(1, 2, 3), Vector(4, 5, 6), Vector(7, 8, 9))),
            Vector(
                Vector(Vector(1, 2, 3), Vector(4, 5, 6), Vector(7, 8, 9)),
                Vector(Vector(10, 11, 12), Vector(13, 14, 15), Vector(16, 17, 18)),
                Vector(Vector(19, 20, 21), Vector(22, 23, 24), Vector(25, 26, 27))).shuffle(0, random))
        assertEquals(
            Vector(
                Vector(Vector(4, 5, 6), Vector(1, 2, 3), Vector(7, 8, 9)),
                Vector(Vector(16, 17, 18), Vector(10, 11, 12), Vector(13, 14, 15)),
                Vector(Vector(22, 23, 24), Vector(19, 20, 21), Vector(25, 26, 27))),
            Vector(
                Vector(Vector(1, 2, 3), Vector(4, 5, 6), Vector(7, 8, 9)),
                Vector(Vector(10, 11, 12), Vector(13, 14, 15), Vector(16, 17, 18)),
                Vector(Vector(19, 20, 21), Vector(22, 23, 24), Vector(25, 26, 27))).shuffle(1, random))
        assertEquals(
            Vector(
                Vector(Vector(1, 2, 3), Vector(4, 6, 5), Vector(7, 9, 8)),
                Vector(Vector(11, 10, 12), Vector(13, 15, 14), Vector(18, 16, 17)),
                Vector(Vector(20, 19, 21), Vector(23, 24, 22), Vector(26, 27, 25))),
            Vector(
                Vector(Vector(1, 2, 3), Vector(4, 5, 6), Vector(7, 8, 9)),
                Vector(Vector(10, 11, 12), Vector(13, 14, 15), Vector(16, 17, 18)),
                Vector(Vector(19, 20, 21), Vector(22, 23, 24), Vector(25, 26, 27))).shuffle(2, random))
    }

    @Test
    fun `insert() returns a row vector with the inserted value`() {
        assertEquals(Vector(5, 1, 2), Vector(1, 2).insert(0, 5))
        assertEquals(Vector(0, 5, 2), Vector(0, 2).insert(1, 5.0))
        assertEquals(Vector(0, 1, 5), Vector(0, 1).insert(2, 5.0))

        // Only row vectors can insert values.
        assertThrows(ShapeException::class.java) {
            Vector(Vector(1, 2), Vector(3, 4)).insert(0, 1.0)
        }
    }

    @Test
    fun `append() returns a row vector with the appended value`() {
        assertEquals(Vector(0, 1, 2), Vector(0, 1).append(2))
        assertEquals(Vector(0, 1, 2), Vector(0, 1).append(2.0))

        // Only row vectors can append values.
        assertThrows(ShapeException::class.java) {
            Vector(Vector(1, 2), Vector(3, 4)).append(1.0)
        }
    }

    @Test
    fun `toDoubleArray() returns a DoubleArray containing all the values`() {
        val array = Vector(Vector(1, 2), Vector(3, 4)).toDoubleArray()
        array.forEachIndexed { index, value ->
            assertEquals(index + 1.0, value)
        }
    }

    @Test
    fun `toMutableVector() returns a MutableVector copy of the vector`() {
        val vector = Vector(Vector(1, 2, 3), Vector(4, 5, 6)).toMutableVector()
        val mutable = vector.toMutableVector()
        assertEquals(vector, mutable)
    }

    @Test
    fun `approxEquals() compares vector values with approximate equality`() {
        assertTrue(Vector(Vector(0.001, 1.001), Vector(1.999, 2.999))
            .approxEquals(Vector(Vector(0, 1), Vector(2, 3)), 0.01))

        assertFalse(Vector(Vector(0.1, 1.001), Vector(1.999, 2.999))
            .approxEquals(Vector(Vector(0, 1), Vector(2, 3)), 0.01))

        assertFalse(Vector(Vector(0.001, 1.001), Vector(1.95, 2.999))
            .approxEquals(Vector(Vector(0, 1), Vector(2, 3)), 0.01))
    }

    @Test
    fun `toString(null) does not truncate large vectors`() {
        assertEquals(
            "[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]",
            Vector(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12).toString(null))
    }

    @Test
    fun `toString() and toString(Int) truncate large vectors`() {
        assertEquals(
            "[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, ..., 12.0]",
            Vector(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12).toString())
        assertEquals(
            "[[1.0, 2.0, ..., 5.0]\n[6.0, 7.0, ..., 10.0]\n...\n[16.0, 17.0, ..., 20.0]]",
            Vector(
                Vector(1, 2, 3, 4, 5),
                Vector(6, 7, 8, 9, 10),
                Vector(11, 12, 13, 14, 15),
                Vector(16, 17, 18, 19, 20)).toString(truncate = 3))
    }
}
