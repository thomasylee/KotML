package kotml.math

/**
 * Matrix stores a collection of values in one or more dimensions. Every
 * matrix has a `dimensions` variable that indicates how many dimensions
 * the matrix has, with the size of each dimension given by `shape`. For
 * example, a 3x4 2-dimensional matrix would have `dimensions` = 2 and
 * `shape` = [3, 4], where 3 is the row count and 4 is the column count.
 *
 * The following operations can be performed on matrices. Note that matrices
 * are immutable, so each operation returns a new Matrix instance.
 * -a = Negates the values of the matrix (e.g. -[1, 2] = [-1, -2])
 * a + b = Entrywise summation (e.g. [1, 2] + [3, 4] = [4, 6])
 * a - b = Entrywise difference (e.g. [3, 0] - [1, 5] = [2, -5])
 * a * b = Entrywise product (e.g. [-1, 2] * [3, 4] = [-3, 8])
 * a / b = Entrywise quotient (e.g. [9, 2] / [3, 1] = [3, 2])
 * a x b = Matrix multiplication product (e.g. [1, -2, 3] x [[4, 1], [0, 0], [-1, 2]] = [1, 7])
 */
class Matrix private constructor(val shape: IntArray) {
    private val scalarValues: DoubleArray
    private val matrixValues: Array<Matrix>
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
            throw ShapeException("Matrices must have at least 1 dimension")
        }

        if (dimensions == 1) {
            scalarValues = DoubleArray(shape[0]) { 0.0 }
            matrixValues = arrayOf()
        } else {
            scalarValues = doubleArrayOf()
            matrixValues = Array<Matrix>(shape[0]) { Matrix(intArrayOf(0)) }
        }
    }

    constructor(vararg values: Double) : this(intArrayOf(values.size)) {
        System.arraycopy(values, 0, scalarValues, 0, values.size)
    }

    constructor(vararg values: Matrix) : this(
        if (values.size == 1) {
            values.first().shape
        } else {
            addDimensionToShape(
                values.size,
                values.elementAtOrElse(0) { Matrix(intArrayOf()) }.shape)
        }
    ) {
        if (values.size == 1) {
            val matrix = values.first()
            if (matrix.dimensions == 1) {
                System.arraycopy(matrix.scalarValues, 0, scalarValues, 0, matrix.shape[0])
            } else {
                (0 until matrix.shape[0]).forEach {
                    matrixValues[it] = matrix.matrixValues[it].clone()
                }
            }
        } else {
            (0 until values.size).forEach {
                matrixValues[it] = values[it]
            }
        }
    }

    operator fun get(index: Int): Matrix {
        if (dimensions == 1) {
            throw ShapeException(
                "Use () instead of [] to access elements of row matrices"
            )
        }
        return matrixValues[index]
    }

    operator fun invoke(index: Int): Double {
        if (dimensions != 1) {
            throw ShapeException(
                "Use [] instead of () to access elements of multidimensional matrices"
            )
        }
        return scalarValues[index]
    }

    operator fun unaryMinus(): Matrix =
        if (dimensions == 1) {
            Matrix(*DoubleArray(shape[0]) { -scalarValues[it] })
        } else {
            Matrix(*Array<Matrix>(shape[0]) { -matrixValues[it] })
        }

    operator fun plus(matrix: Matrix): Matrix {
        validateShapesMatch(matrix)
        if (dimensions == 1) {
            return Matrix(*DoubleArray(shape[0]) {
                scalarValues[it] + matrix.scalarValues[it]
            })
        }
        return Matrix(*Array<Matrix>(shape[0]) {
            matrixValues[it] + matrix.matrixValues[it]
        })
    }

    operator fun minus(matrix: Matrix): Matrix {
        validateShapesMatch(matrix)
        if (dimensions == 1) {
            return Matrix(*DoubleArray(shape[0]) {
                scalarValues[it] - matrix.scalarValues[it]
            })
        }
        return Matrix(*Array<Matrix>(shape[0]) {
            matrixValues[it] - matrix.matrixValues[it]
        })
    }

    operator fun times(matrix: Matrix): Matrix {
        validateShapesMatch(matrix)
        if (dimensions == 1) {
            return Matrix(*DoubleArray(shape[0]) {
                scalarValues[it] * matrix.scalarValues[it]
            })
        }
        return Matrix(*Array<Matrix>(shape[0]) {
            matrixValues[it] * matrix.matrixValues[it]
        })
    }

    operator fun div(matrix: Matrix): Matrix {
        validateShapesMatch(matrix)
        if (dimensions == 1) {
            return Matrix(*DoubleArray(shape[0]) {
                scalarValues[it] / matrix.scalarValues[it]
            })
        }
        return Matrix(*Array<Matrix>(shape[0]) {
            matrixValues[it] / matrix.matrixValues[it]
        })
    }

    /**
     * Returns the matrix multiplication product of this matrix x the
     * provided matrix.
     * @param matrix Matrix to matrix multiply with this matrix
     * @return the matrix multiplication product
     */
    infix fun x(other: Matrix): Matrix {
        if (dimensions > 2 || other.dimensions > 2) {
            throw ShapeException("Matrix multiplication can only be performed on 2-dimensional matrices")
        }
        if ((dimensions == 1 && other.dimensions == 1 && shape[0] != 1) ||
                (dimensions == 1 && shape[0] != other.shape[0]) ||
                (dimensions == 2 && shape[1] != other.shape[0])) {
            throw ShapeException(
                "Matrix multiplication requires the column count of the first matrix to equal the row count of the second matrix"
            )
        }

        if (dimensions == 1) {
            if (other.dimensions == 1) {
                return Matrix(scalarValues[0] * other.scalarValues[0])
            } else {
                return Matrix(*DoubleArray(other.shape[1]) { col ->
                    (0 until shape[0]).fold(0.0) { acc, offset ->
                        acc + scalarValues[offset] * other[offset](col)
                    }
                })
            }
        }
        return Matrix(*Array<Matrix>(shape[0]) { row ->
            Matrix(*DoubleArray(other.shape[1]) { col ->
                (0 until shape[1]).fold(0.0) { acc, offset ->
                    acc + matrixValues[row](offset) * other[offset](col)
                }
            })
        })
    }

    fun transpose(): Matrix {
        if (dimensions > 2) {
            throw ShapeException(
                "Only matrices with 1 or 2 dimensions can be transposed"
            )
        }
        if (dimensions == 1) {
            return Matrix(*Array<Matrix>(shape[0]) { index ->
                Matrix(scalarValues[index])
            })
        } else if (shape[0] == 1) {
            return Matrix(*DoubleArray(shape[1]) { index ->
                matrixValues[index](0)
            })
        } else {
            return Matrix(*Array<Matrix>(shape[1]) { row ->
                Matrix(*DoubleArray(shape[0]) { col ->
                    matrixValues[col](row)
                })
            })
        }
    }

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

    fun clone(): Matrix =
        if (dimensions == 1) {
            Matrix(*scalarValues.clone())
        } else {
            Matrix(*Array<Matrix>(shape[0]) { matrixValues[it].clone() })
        }

    override fun equals(other: Any?): Boolean =
        if (other != null && other is Matrix) {
            shapeEquals(other.shape) &&
                (0 until scalarValues.size).all {
                    scalarValues[it] == other.scalarValues[it]
                } &&
                (0 until matrixValues.size).all {
                    matrixValues[it] == other.matrixValues[it]
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
            "[" + matrixValues.joinToString("\n") + "]"
        }

    private fun validateSquareMatrix() {
        if (dimensions > 2 || (dimensions == 1 && shape[0] != 1) || shape[0] != shape[1]) {
            throw ShapeException(
                "Only 2-dimensional square matrices have defined inverses"
            )
        }
    }

    private fun validateShapesMatch(matrix: Matrix) {
        if (!shapeEquals(matrix.shape)) {
            throw ShapeException(
                "Matrix sizes do not match: ${shapeToString(shape)} and ${shapeToString(matrix.shape)}"
            )
        }
    }
}
