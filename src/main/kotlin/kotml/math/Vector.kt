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
class Vector private constructor(val shape: IntArray) {
    private val scalarValues: DoubleArray
    private val vectorValues: Array<Vector>
    val dimensions: Int

    companion object {
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
        private fun shapeToString(shape: IntArray): String =
            "[" + shape.joinToString(", ") + "]"
    }

    init {
        dimensions = shape.size
        if (dimensions == 0) {
            throw ShapeException("Vectors must have at least 1 dimension")
        }

        if (dimensions == 1) {
            scalarValues = DoubleArray(shape[0]) { 0.0 }
            vectorValues = arrayOf()
        } else {
            scalarValues = doubleArrayOf()
            vectorValues = Array<Vector>(shape[0]) { Vector(intArrayOf(0)) }
        }
    }

    constructor(vararg values: Double) : this(intArrayOf(values.size)) {
        System.arraycopy(values, 0, scalarValues, 0, values.size)
    }

    constructor(vararg values: Vector) : this(
        if (values.size == 1) {
            values.first().shape
        } else {
            addDimensionToShape(
                values.size,
                values.elementAtOrElse(0) { Vector(intArrayOf()) }.shape)
        }
    ) {
        if (values.size == 1) {
            val vector = values.first()
            if (vector.dimensions == 1) {
                System.arraycopy(vector.scalarValues, 0, scalarValues, 0, vector.shape[0])
            } else {
                (0 until vector.shape[0]).forEach {
                    vectorValues[it] = vector.vectorValues[it].clone()
                }
            }
        } else {
            (0 until values.size).forEach {
                vectorValues[it] = values[it]
            }
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

    fun transpose(): Vector {
        if (dimensions > 2) {
            throw ShapeException(
                "Only vectors with 1 or 2 dimensions can be transposed"
            )
        }
        if (dimensions == 1) {
            return Vector(*Array<Vector>(shape[0]) { index ->
                Vector(scalarValues[index])
            })
        } else if (shape[0] == 1) {
            return Vector(*DoubleArray(shape[1]) { index ->
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

    fun clone(): Vector =
        if (dimensions == 1) {
            Vector(*scalarValues.clone())
        } else {
            Vector(*Array<Vector>(shape[0]) { vectorValues[it].clone() })
        }

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

    fun shapeEquals(otherShape: IntArray): Boolean =
        dimensions == otherShape.size &&
            (0 until dimensions).all { shape[it] == otherShape[it] }

    override fun toString(): String =
        if (dimensions == 1) {
            "[" + scalarValues.joinToString(", ") + "]"
        } else {
            "[" + vectorValues.joinToString("\n") + "]"
        }

    private fun validateShapesMatch(vector: Vector) {
        if (!shapeEquals(vector.shape)) {
            throw ShapeException(
                "Vector sizes do not match: ${shapeToString(shape)} and ${shapeToString(vector.shape)}"
            )
        }
    }
}
