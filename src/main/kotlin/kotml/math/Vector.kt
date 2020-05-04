package kotml.math

/**
 * Vector stores a collection of values in one or more dimensions. Every
 * vector has a `dimensions` variable that indicates how many dimensions
 * the vector has, with the size of each dimension given by `shape`. For
 * example, a 3x4 2-dimensional vector would have `dimensions` = 2 and
 * `shape` = [3, 4], where 3 is the row count and 4 is the column count.
 *
 * The following operations can be performed on vectors. Note that vectors
 * are immutable, so each operation returns a new Vector instance.
 * -a = Negates the values of the vector (e.g. -[1, 2] = [-1, -2])
 * a + b = Entrywise summation (e.g. [1, 2] + [3, 4] = [4, 6])
 * a - b = Entrywise difference (e.g. [3, 0] - [1, 5] = [2, -5])
 * a * b = Entrywise product (e.g. [-1, 2] * [3, 4] = [-3, 8])
 * a / b = Entrywise quotient (e.g. [9, 2] / [3, 1] = [3, 2])
 * a x b = Matrix multiplication product (e.g. [1, -2, 3] x [[4, 1], [0, 0], [-1, 2]] = [1, 7])
 */
class Vector private constructor(initIndex: Int, val shape: IntArray, mapValues: (Int) -> Double) {
    private val scalarValues: DoubleArray
    private val vectorValues: Array<Vector>
    val dimensions: Int

    companion object {
        @JvmStatic
        fun zeros(shape: IntArray): Vector = Vector(shape) { 0.0 }

        @JvmStatic
        private fun addDimensionToShape(dimension: Int, shape: IntArray): IntArray =
            IntArray(shape.size + 1) { index ->
                if (index == 0)
                    dimension
                else
                    shape[index - 1]
            }

        @JvmStatic
        private fun removeDimensionFromShape(shape: IntArray): IntArray =
            IntArray(shape.size - 1) { shape[it + 1] }

        @JvmStatic
        fun shapeToString(shape: IntArray): String =
            "[" + shape.joinToString(", ") + "]"
    }

    init {
        dimensions = shape.size
        if (dimensions == 0) {
            throw ShapeException("Vectors must have at least 1 dimension")
        }

        if (dimensions == 1) {
            scalarValues = DoubleArray(shape[0]) { mapValues(initIndex + it) }
            vectorValues = arrayOf()
        } else {
            scalarValues = doubleArrayOf()
            val subVectorShape = removeDimensionFromShape(shape)
            val scalarsPerVector = subVectorShape.fold(1) { acc, dim ->
                acc * dim
            }
            vectorValues = Array<Vector>(shape[0]) { index ->
                Vector(initIndex + index * scalarsPerVector, subVectorShape, mapValues)
            }
        }
    }

    constructor(shape: IntArray, mapValues: (Int) -> Double) : this(0, shape, mapValues)

    constructor(vararg values: Double) : this(0, intArrayOf(values.size), {
        values[it]
    })

    constructor(vararg values: Int) : this(0, intArrayOf(values.size), {
        values[it].toDouble()
    })

    constructor(vararg values: Vector) : this(
        0,
        addDimensionToShape(
            values.size,
            values.elementAtOrNull(0)?.shape ?: intArrayOf()),
        { 0.0 }
    ) {
        (0 until values.size).forEach {
            vectorValues[it] = values[it]
        }
    }

    operator fun get(index: Int): Vector {
        if (dimensions == 1) {
            throw ShapeException(
                "Use () instead of [] to access elements of row vectors"
            )
        }
        return vectorValues[index]
    }

    operator fun invoke(index: Int): Double {
        if (dimensions != 1) {
            throw ShapeException(
                "Use [] instead of () to access elements of multidimensional vectors"
            )
        }
        return scalarValues[index]
    }

    operator fun unaryMinus(): Vector =
        if (dimensions == 1) {
            Vector(*DoubleArray(shape[0]) { -scalarValues[it] })
        } else {
            Vector(*Array<Vector>(shape[0]) { -vectorValues[it] })
        }

    operator fun plus(vector: Vector): Vector {
        validateShapesMatch(vector)
        if (dimensions == 1) {
            return Vector(*DoubleArray(shape[0]) {
                scalarValues[it] + vector.scalarValues[it]
            })
        }
        return Vector(*Array<Vector>(shape[0]) {
            vectorValues[it] + vector.vectorValues[it]
        })
    }

    operator fun plus(value: Double): Vector = map { it + value }

    operator fun plus(value: Int): Vector = map { it + value }

    operator fun minus(vector: Vector): Vector {
        validateShapesMatch(vector)
        if (dimensions == 1) {
            return Vector(*DoubleArray(shape[0]) {
                scalarValues[it] - vector.scalarValues[it]
            })
        }
        return Vector(*Array<Vector>(shape[0]) {
            vectorValues[it] - vector.vectorValues[it]
        })
    }

    operator fun minus(value: Double): Vector = map { it - value }

    operator fun minus(value: Int): Vector = map { it - value }

    operator fun times(vector: Vector): Vector {
        validateShapesMatch(vector)
        if (dimensions == 1) {
            return Vector(*DoubleArray(shape[0]) {
                scalarValues[it] * vector.scalarValues[it]
            })
        }
        return Vector(*Array<Vector>(shape[0]) {
            vectorValues[it] * vector.vectorValues[it]
        })
    }

    operator fun times(value: Double): Vector = map { it * value }

    operator fun times(value: Int): Vector = map { it * value }

    operator fun div(vector: Vector): Vector {
        validateShapesMatch(vector)
        if (dimensions == 1) {
            return Vector(*DoubleArray(shape[0]) {
                scalarValues[it] / vector.scalarValues[it]
            })
        }
        return Vector(*Array<Vector>(shape[0]) {
            vectorValues[it] / vector.vectorValues[it]
        })
    }

    operator fun div(value: Double): Vector = map { it / value }

    operator fun div(value: Int): Vector = map { it / value }

    /**
     * Returns the matrix multiplication product of this vector x the
     * provided vector.
     * @param vector Vector to matrix multiply with this vector
     * @return the matrix multiplication product
     */
    infix fun x(other: Vector): Vector {
        if (dimensions > 2 || other.dimensions > 2) {
            throw ShapeException("Matrix multiplication can only be performed on 2-dimensional vectors")
        }
        if ((dimensions == 1 && other.dimensions == 1 && shape[0] != 1) ||
                (dimensions == 1 && shape[0] != other.shape[0]) ||
                (dimensions == 2 && shape[1] != other.shape[0])) {
            throw ShapeException(
                "Matrix multiplication requires the column count of the first vector to equal the row count of the second vector"
            )
        }

        if (dimensions == 1) {
            if (other.dimensions == 1) {
                return Vector(scalarValues[0] * other.scalarValues[0])
            } else {
                return Vector(*DoubleArray(other.shape[1]) { col ->
                    (0 until shape[0]).fold(0.0) { acc, offset ->
                        acc + scalarValues[offset] * other[offset](col)
                    }
                })
            }
        }
        return Vector(*Array<Vector>(shape[0]) { row ->
            Vector(*DoubleArray(other.shape[1]) { col ->
                (0 until shape[1]).fold(0.0) { acc, offset ->
                    acc + vectorValues[row](offset) * other[offset](col)
                }
            })
        })
    }

    private fun mapIndexed(startIndex: Int, fn: (Int, Double) -> Double): Vector =
        if (dimensions == 1) {
            Vector(*DoubleArray(shape[0]) { fn(startIndex + it, scalarValues[it]) })
        } else {
            Vector(*Array<Vector>(shape[0]) {
                vectorValues[it].mapIndexed(
                    it * removeDimensionFromShape(shape).fold(1) { acc, dim ->
                        acc * dim
                    },
                    fn
                )
            })
        }

    /**
     * Returns a vector with values mapped to the given function `fn`.
     * @param fn function to apply to each element of the vector
     * @return the mapped vector
     */
    fun mapIndexed(fn: (Int, Double) -> Double): Vector = mapIndexed(0, fn)

    /**
     * Returns a vector with values mapped to the given function `fn`.
     * @param fn function to apply to each element of the vector
     * @return the mapped vector
     */
    fun map(fn: (Double) -> Double): Vector = mapIndexed { _, value ->
        fn(value)
    }

    /**
     * Returns the fold along the specified axis using the function `fn`.
     * @param initial the initial value used in the fold operation
     * @param axis the axis used to fold on
     * @param fn the function applied on the accumulator and individual value
     * @return vector of the results
     */
    fun fold(initial: Double, axis: Int = 0, fn: (Double, Double) -> Double): Vector {
        if (axis > dimensions)
            throw ShapeException("Axis $axis must be less than or equal to the  number of dimensions $dimensions")

        if (dimensions == 1) {
            return Vector(scalarValues.fold(initial, fn))
        }

        // Example: [[1,2,3],[4,5,6]].sum(axis=0) = [5,7,9]
        if (dimensions == 2 && axis == 0) {
            return Vector(*DoubleArray(shape[1]) { col ->
                (0 until shape[0]).fold(initial) { acc, row ->
                    fn(acc, vectorValues[row](col))
                }
            })
        }

        // Example: [[1,2,3],[4,5,6]].sum(axis=2) = [6, 15]
        if (dimensions == 2 && axis == 1) {
            return Vector(*DoubleArray(shape[0]) { row ->
                (0 until shape[1]).fold(initial) { acc, col ->
                    fn(acc, vectorValues[row](col))
                }
            })
        }

        return this
    }

    /**
     * Returns the sum of elements along a particular axis.
     * @param axis the axis used to calculate the summation vector
     */
    fun sum(axis: Int = 0): Vector = fold(0.0, axis) {
        acc, value -> acc + value
    }

    /**
     * Returns the product of elements along a particular axis.
     * @param axis the axis used to calculate the product vector
     */
    fun product(axis: Int = 0): Vector = fold(1.0, axis) {
        acc, value -> acc * value
    }

    /**
     * Returns the dot product of this vector with `other`.
     * @param other the vector to dot product with this vector
     * @return dot product of this vector and `other`
     */
    infix fun dot(other: Vector): Double = (this * other).sum()(0)

    /**
     * Returns the transposition of this vector.
     * @return vector that is the transpose of this vector
     */
    fun transpose(): Vector {
        if (dimensions > 2) {
            throw ShapeException(
                "Only vectors with 1 or 2 dimensions can be transposed"
            )
        }
        if (dimensions == 1 && shape[0] == 1) {
            return this
        } else if (dimensions == 1) {
            return Vector(*Array<Vector>(shape[0]) { index ->
                Vector(scalarValues[index])
            })
        } else if (shape[1] == 1) {
            return Vector(*DoubleArray(shape[0]) { index ->
                vectorValues[index](0)
            })
        } else {
            return Vector(*Array<Vector>(shape[1]) { row ->
                Vector(*DoubleArray(shape[0]) { col ->
                    vectorValues[col](row)
                })
            })
        }
    }

    /**
     * Returns the determinant of this vector.
     * @return determinant of the vector
     */
    fun det(): Double {
        validateSquareMatrix()

        if (dimensions == 1)
            throw ShapeException("1x1 matrices do not have inverses")

        if (shape[0] == 2)
            return this[0](0) * this[1](1) - this[0](1) * this[1](0)

        val toAdd = (0 until shape[0]).fold(0.0) { sumAcc, topIndex ->
            sumAcc + (0 until shape[0]).fold(1.0) { productAcc, offset ->
                val col = (topIndex + offset) % shape[0]
                productAcc * this[offset](col)
            }
        }
        val toSubtract = (0 until shape[0]).fold(0.0) { diffAcc, topIndex ->
            diffAcc + ((shape[0] - 1) downTo 0).fold(1.0) { productAcc, offset ->
                val col = (shape[0] + topIndex - offset) % shape[0]
                productAcc * this[offset](col)
            }
        }
        return toAdd - toSubtract
    }

    /**
     * Returns true if `other` is a Vector with the same shape and values
     * as this vector.
     * @param other vector to check equality against
     * @return true if the vectors have equivalent values, false otherwise
     */
    override fun equals(other: Any?): Boolean =
        if (other != null && other is Vector) {
            shapeEquals(other.shape) &&
                (0 until scalarValues.size).all {
                    scalarValues[it] == other.scalarValues[it]
                } &&
                (0 until vectorValues.size).all {
                    vectorValues[it] == other.vectorValues[it]
                }
        } else {
            false
        }

    /**
     * Returns true if this vector's shape equals the given shape.
     * @param otherShape shape to compare to this vector's shape
     * @return true if the shapes have the same values, false otherwise
     */
    fun shapeEquals(otherShape: IntArray): Boolean =
        dimensions == otherShape.size &&
            (0 until dimensions).all { shape[it] == otherShape[it] }

    /**
     * Returns a human-readable String representation of this vector.
     * @return String representation of the vector
     */
    override fun toString(): String =
        if (dimensions == 1) {
            "[" + scalarValues.joinToString(", ") + "]"
        } else {
            "[" + vectorValues.joinToString("\n") + "]"
        }

    private fun validateSquareMatrix() {
        if (dimensions > 2 || (dimensions == 1 && shape[0] != 1) || shape[0] != shape[1]) {
            throw ShapeException(
                "Only 2-dimensional square matrices have defined inverses"
            )
        }
    }

    private fun validateShapesMatch(vector: Vector) {
        if (!shapeEquals(vector.shape)) {
            throw ShapeException(
                "Vector sizes do not match: ${shapeToString(shape)} and ${shapeToString(vector.shape)}"
            )
        }
    }
}
