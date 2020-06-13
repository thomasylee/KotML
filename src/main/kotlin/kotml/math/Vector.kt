package kotml.math

import kotlin.math.abs
import kotlin.random.Random

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
 * a dot b = Dot product (e.g. [1, 2] dot [3, 4] = 1*3+2*4 = 11
 * a x b = Matrix multiplication product (e.g. [1, -2, 3] x [[4, 1], [0, 0], [-1, 2]] = [1, 7])
 * a.transpose() = Matrix transpose (e.g. [1, 2].transpose() = [[1], [2]])
 * a.det() = Matrix determinant (e.g. [[1, 2], [3, 4]].det() = -2)
 */
open class Vector private constructor(
    initIndex: Byte,
    val shape: IntArray,
    mapValues: (Int) -> Double
) {
    val dimensions: Int

    // Unfortunately, these need to be internal for MutableVector to access them.
    internal val scalarValues: DoubleArray
    internal val vectorValues: Array<Vector>

    companion object {
        // Minimum matrix dimension that will use Strassen for matrix
        // multiplication.
        var minStrassenSize = 256

        @JvmStatic
        fun zeros(vararg shape: Int): Vector = Vector(*shape) { 0.0 }

        @JvmStatic
        fun ofVectors(numVectors: Int, mapVectors: (Int) -> Vector): Vector =
            Vector(*Array<Vector>(numVectors) { index ->
                mapVectors(index)
            })

        @JvmStatic
        protected fun addDimensionToShape(dimension: Int, shape: IntArray): IntArray =
            IntArray(shape.size + 1) { index ->
                if (index == 0)
                    dimension
                else
                    shape[index - 1]
            }

        @JvmStatic
        protected fun removeDimensionFromShape(shape: IntArray): IntArray =
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
                Vector(
                    initIndex = (initIndex + index * scalarsPerVector).toByte(),
                    shape = subVectorShape,
                    mapValues = mapValues
                )
            }
        }
    }

    constructor(vararg shape: Int, mapValues: (Int) -> Double) : this(
        initIndex = 0.toByte(),
        shape = shape,
        mapValues = mapValues
    )

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

    operator fun get(vararg indices: Int): Double {
        if (indices.size != dimensions) {
            throw ShapeException(
                "Number of indices [${indices.joinToString(", ")}] does not match vector dimensions $dimensions"
            )
        }
        var vector = this
        (0 until (indices.size - 1)).forEach {
            vector = vector.vectorValues[indices[it]]
        }
        return vector.scalarValues[indices.last()]
    }

    operator fun invoke(vararg indices: Int): Vector {
        if (indices.size >= dimensions) {
            throw ShapeException(
                "Number of indices [${indices.joinToString(", ")}] must be less than the number of vector dimensions $dimensions"
            )
        }
        var vector = this
        indices.forEach {
            vector = vector.vectorValues[it]
        }
        return vector
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

    private fun strassenMatmul(other: Vector): Vector {
        val first = Array<DoubleArray>(shape[0]) { row ->
            DoubleArray(shape[1]) { col ->
                this[row, col]
            }
        }
        val second = Array<DoubleArray>(other.shape[0]) { row ->
            DoubleArray(other.shape[1]) { col ->
                other[row, col]
            }
        }
        val product = strassenMatmul(first, second)
        return Vector.ofVectors(product.size) { row ->
            Vector(other.shape[1]) { col ->
                product[row][col]
            }
        }
    }

    private fun strassenMatmul(first: Array<DoubleArray>, second: Array<DoubleArray>): Array<DoubleArray> {
        val size = first.size

        if (size < minStrassenSize)
            return ikjMatmul(first, second)

        val newSize = size / 2

        val a11 = Array<DoubleArray>(newSize) { DoubleArray(newSize) }
        val a12 = Array<DoubleArray>(newSize) { DoubleArray(newSize) }
        val a21 = Array<DoubleArray>(newSize) { DoubleArray(newSize) }
        val a22 = Array<DoubleArray>(newSize) { DoubleArray(newSize) }

        val b11 = Array<DoubleArray>(newSize) { DoubleArray(newSize) }
        val b12 = Array<DoubleArray>(newSize) { DoubleArray(newSize) }
        val b21 = Array<DoubleArray>(newSize) { DoubleArray(newSize) }
        val b22 = Array<DoubleArray>(newSize) { DoubleArray(newSize) }

        var aResult = Array<DoubleArray>(newSize) { DoubleArray(newSize) }
        var bResult = Array<DoubleArray>(newSize) { DoubleArray(newSize) }

        (0 until newSize).forEach { i ->
            (0 until newSize).forEach { j ->
                a11[i][j] = first[i][j]
                a12[i][j] = first[i][j + newSize]
                a21[i][j] = first[i + newSize][j]
                a22[i][j] = first[i + newSize][j + newSize]

                b11[i][j] = second[i][j]
                b12[i][j] = second[i][j + newSize]
                b21[i][j] = second[i + newSize][j]
                b22[i][j] = second[i + newSize][j + newSize]
            }
        }

        add(a11, a22, aResult)
        add(b11, b22, bResult)
        val p1 = strassenMatmul(aResult, bResult)

        add(a21, a22, aResult)
        val p2 = strassenMatmul(aResult, b11)

        subtract(b12, b22, bResult)
        val p3 = strassenMatmul(a11, bResult)

        subtract(b21, b11, bResult)
        val p4 = strassenMatmul(a22, bResult)

        add(a11, a12, aResult)
        val p5 = strassenMatmul(aResult, b22)

        subtract(a21, a11, aResult)
        add(b11, b12, bResult)
        val p6 = strassenMatmul(aResult, bResult)

        subtract(a12, a22, aResult)
        add(b21, b22, bResult)
        val p7 = strassenMatmul(aResult, bResult)

        var c12 = Array<DoubleArray>(newSize) { DoubleArray(newSize) }
        add(p3, p5, c12)

        var c21 = Array<DoubleArray>(newSize) { DoubleArray(newSize) }
        add(p2, p4, c21)

        add(p1, p4, aResult)
        add(aResult, p7, bResult)
        var c11 = Array<DoubleArray>(newSize) { DoubleArray(newSize) }
        subtract(bResult, p5, c21)

        add(p1, p3, aResult)
        add(aResult, p6, bResult)
        var c22 = Array<DoubleArray>(newSize) { DoubleArray(newSize) }
        subtract(bResult, p2, c22)

        (0 until newSize).forEach { i ->
            (0 until newSize).forEach { j ->
                first[i][j] = c11[i][j]
                first[i][j + newSize] = c12[i][j]
                first[i + newSize][j] = c21[i][j]
                first[i + newSize][j + newSize] = c22[i][j]
            }
        }
        return first
    }

    private fun add(first: Array<DoubleArray>, second: Array<DoubleArray>, product: Array<DoubleArray>) {
        (0 until product.size).forEach { i ->
            (0 until product[i].size).forEach { j ->
                product[i][j] = first[i][j] + second[i][j]
            }
        }
    }

    private fun subtract(first: Array<DoubleArray>, second: Array<DoubleArray>, product: Array<DoubleArray>) {
        (0 until product.size).forEach { i ->
            (0 until product[i].size).forEach { j ->
                product[i][j] = first[i][j] - second[i][j]
            }
        }
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
                        acc + scalarValues[offset] * other(offset)[col]
                    }
                })
            }
        }
        // return ikjMatmul(other)
        return strassenMatmul(other)
    }

    fun ikjMatmul(other: Vector): Vector {
        val first = Array<DoubleArray>(shape[0]) { row ->
            DoubleArray(shape[1]) { col ->
                this[row, col]
            }
        }
        val second = Array<DoubleArray>(other.shape[0]) { row ->
            DoubleArray(other.shape[1]) { col ->
                other[row, col]
            }
        }
        val product = ikjMatmul(first, second)
        return Vector.ofVectors(product.size) { row ->
            Vector(other.shape[1]) { col ->
                product[row][col]
            }
        }
    }

    fun ikjMatmul(first: Array<DoubleArray>, second: Array<DoubleArray>): Array<DoubleArray> {
        val product = Array<DoubleArray>(first.size) {
            DoubleArray(second[0].size)
        }
        (0 until first.size).forEach { i ->
            (0 until second.size).forEach { k ->
                (0 until second[0].size).forEach { j ->
                    product[i][j] += first[i][k] * second[k][j]
                }
            }
        }
        return product
    }

    private fun forEachIndexed(startIndex: Int, fn: (Int, Double) -> Unit) {
        if (dimensions == 1) {
            scalarValues.forEachIndexed { index, value ->
                fn(startIndex + index, value)
            }
            return
        }

        val subShape = removeDimensionFromShape(shape)
        val valuesPerVector = subShape.fold(1) { acc, value -> acc * value }
        vectorValues.forEachIndexed { index, vector ->
            vector.forEachIndexed(startIndex + index * valuesPerVector, fn)
        }
    }

    /**
     * Iterates over all scalar values in the vector.
     * @param fn function to invoke on each value
     */
    fun forEachIndexed(fn: (Int, Double) -> Unit) = forEachIndexed(0, fn)

    /**
     * Iterates over all scalar values in the vector.
     * @param fn function to invoke on each value
     */
    fun forEach(fn: (Double) -> Unit) = forEachIndexed { _, value ->
        fn(value)
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
     * @param initial initial value used in the fold operation
     * @param initIndex initial index
     * @param axis axis used to fold on
     * @param fn function applied on the indices, accumulator, and individual value
     * @return vector of the results
     */
    private fun foldIndexed(initial: Double, initIndex: Int, axis: Int = 0, fn: (Int, Double, Double) -> Double): Vector {
        if (axis > dimensions)
            throw ShapeException("Axis $axis must be less than or equal to the  number of dimensions $dimensions")

        if (dimensions == 1) {
            return Vector(scalarValues.foldIndexed(initial) { index, acc, value ->
                fn(initIndex + index, acc, value)
            })
        }

        // Example: [[1,2,3],[4,5,6]].sum(axis=0) = [5,7,9]
        if (dimensions == 2 && axis == 0) {
            return Vector(shape[1]) { scalarIndex ->
                vectorValues.foldIndexed(initial) { vectorIndex, acc, vector ->
                    fn(initIndex + vectorIndex * shape[1] + scalarIndex, acc, vector[scalarIndex])
                }
            }
        }

        // Example: [[1,2,3],[4,5,6]].sum(axis=1) = [6, 15]
        if (dimensions == 2 && axis == 1) {
            return Vector(*DoubleArray(shape[0]) { vectorIndex ->
                (0 until shape[1]).foldIndexed(initial) { scalarIndex, acc, col ->
                    fn(initIndex + vectorIndex * shape[1] + scalarIndex, acc, vectorValues[vectorIndex][col])
                }
            })
        }

        throw ShapeException("Folding is currently only supported for vectors with 1 or 2 dimensions")
    }

    /**
     * Returns the fold along the specified axis using the function `fn`.
     * @param initial the initial value used in the fold operation
     * @param axis the axis used to fold on
     * @param fn the function applied on the indices, accumulator, and individual values
     * @return vector of the results
     */
    fun foldIndexed(initial: Double, axis: Int = 0, fn: (Int, Double, Double) -> Double): Vector =
        foldIndexed(initial, 0, axis, fn)

    /**
     * Returns the fold along the specified axis using the function `fn`.
     * @param initial the initial value used in the fold operation
     * @param axis the axis used to fold on
     * @param fn the function applied on the accumulator and individual value
     * @return vector of the results
     */
    fun fold(initial: Double, axis: Int = 0, fn: (Double, Double) -> Double): Vector =
        foldIndexed(initial, axis) { _, acc, value -> fn(acc, value) }

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
     * Returns the maximum values along a particular axis.
     * @return vector of maximum values
     */
    fun max(axis: Int = 0): Vector = fold(-Double.MAX_VALUE, axis = axis) { max, value ->
        if (value > max) value else max
    }

    /**
     * Returns the minimum values along a particular axis.
     * @return vector of minimum values
     */
    fun min(axis: Int = 0): Vector = fold(Double.MAX_VALUE, axis = axis) { min, value ->
        if (value < min) value else min
    }

    /**
     * Returns the index of the largest value in the Vector. If several
     * indices have the largest value, one index is randomly chosen.
     * @return index of largest value, with ties broken randomly
     */
    fun argmax(random: Random = Random): Int {
        var maxValue = -Double.MAX_VALUE
        val maxIndices = mutableListOf<Int>()
        forEachIndexed { index, value ->
            if (value > maxValue) {
                maxIndices.clear()
                maxIndices.add(index)
                maxValue = value
            } else if (value == maxValue) {
                maxIndices.add(index)
            }
        }
        return maxIndices.random(random)
    }

    /**
     * Returns the dot product of this vector with `other`.
     * @param other the vector to dot product with this vector
     * @return dot product of this vector and `other`
     */
    infix fun dot(other: Vector): Double = (this * other).sum()[0]

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
                vectorValues[index][0]
            })
        } else {
            return Vector(*Array<Vector>(shape[1]) { row ->
                Vector(*DoubleArray(shape[0]) { col ->
                    vectorValues[col][row]
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
            throw ShapeException("1x1 matrices do not have determinants")

        if (shape[0] == 2)
            return this[0, 0] * this[1, 1] - this[0, 1] * this[1, 0]

        val toAdd = (0 until shape[0]).fold(0.0) { sumAcc, topIndex ->
            sumAcc + (0 until shape[0]).fold(1.0) { productAcc, offset ->
                val col = (topIndex + offset) % shape[0]
                productAcc * this[offset, col]
            }
        }
        val toSubtract = (0 until shape[0]).fold(0.0) { diffAcc, topIndex ->
            diffAcc + ((shape[0] - 1) downTo 0).fold(1.0) { productAcc, offset ->
                val col = (shape[0] + topIndex - offset) % shape[0]
                productAcc * this[offset, col]
            }
        }
        return toAdd - toSubtract
    }

    /**
     * Returns the inverse of the matrix represented by the vector.
     * @return matrix inverse
     */
    fun inverse(): Vector {
        validateSquareMatrix()

        if (dimensions == 1)
            return Vector(1.0 / scalarValues[0])

        if (shape[0] == 2)
            return Vector(
                Vector(this[1, 1], -this[0, 1]),
                Vector(-this[1, 0], this[0, 0])) / det()

        val determinant = det()
        if (determinant == 0.0)
            throw ShapeException("Inverse matrix does not exist")

        return Vector(shape[0], shape[0]) { index ->
            val factor = -2.0 * (index % 2) + 1.0
            val row = index / shape[0]
            val col = index - row * shape[0]
            factor * subMatrix(row, col).det()
        }.transpose() / determinant
    }

    /**
     * Returns the submatrix excluding a specific row and column. This
     * method can only be invoked on square matrices with at least 2 elements
     * per row and column.
     * @param row row to exclude
     * @param col column to exclude
     * @return submatrix excluding the row and column
     */
    fun subMatrix(row: Int, col: Int): Vector {
        validateSquareMatrix()

        if (dimensions == 1)
            throw ShapeException("1x1 matrices do not have matrix minors")

        if (row < 0 || row >= shape[0] || col < 0 || col >= shape[0])
            throw ShapeException("Invalid row and column for matrix minor")

        var rowOffset = 0
        var colOffset = 0
        var prevRow = -1
        val newDim = shape[0] - 1
        return Vector(newDim, newDim) { index ->
            val curRow = index / newDim
            val curCol = index - curRow * newDim
            if (curRow != prevRow) {
                prevRow = curRow
                colOffset = 0
            }
            if (curRow == row)
                rowOffset = 1
            if (curCol == col)
                colOffset = 1
            this[curRow + rowOffset, curCol + colOffset]
        }
    }

    /**
     * Returns a row vector containing the same scalar values.
     * @return flattened row vector
     */
    fun flatten(): Vector = Vector(*toDoubleArray())

    /**
     * Returns the scalar values of the vector across all dimensions. For
     * example, [[1, 2], [3, 4]].toDoubleArray() returns [1, 2, 3, 4].
     * @return DoubleArray containing all scalar values in the vector
     */
    fun toDoubleArray(): DoubleArray {
        if (dimensions == 1)
            return DoubleArray(shape[0]) { scalarValues[it] }

        val valuesPerVector = removeDimensionFromShape(shape).fold(1) { acc, dim ->
            acc * dim
        }
        val valuesOfVectors = vectorValues.map { it.toDoubleArray() }
        return DoubleArray(shape[0] * valuesPerVector) { index ->
            val vectorIndex = index / valuesPerVector
            valuesOfVectors[vectorIndex][index - vectorIndex * valuesPerVector]
        }
    }

    /**
     * Returns a new vector with the value inserted at the specified index.
     * @param index index where the value should be inserted
     * @param value value to insert
     * @return new vector with the inserted value
     */
    fun insert(index: Int, value: Double): Vector {
        if (dimensions != 1)
            throw ShapeException("Only row vectors can insert values")
        var found = 0
        return Vector(shape[0] + 1) {
            if (it == index) {
                found = 1
                value
            } else {
                scalarValues[it - found]
            }
        }
    }

    /**
     * Returns a new vector with the value inserted at the specified index.
     * @param index index where the value should be inserted
     * @param value value to insert
     * @return new vector with the inserted value
     */
    fun insert(index: Int, value: Int): Vector = insert(index, value.toDouble())

    /**
     * Returns a new vector with the value inserted at the specified index.
     * @param index index where the value should be inserted
     * @param value value to insert
     * @return new vector with the inserted value
     */
    fun append(value: Double): Vector = insert(shape[0], value.toDouble())

    /**
     * Returns a new vector with the value inserted at the specified index.
     * @param index index where the value should be inserted
     * @param value value to insert
     * @return new vector with the inserted value
     */
    fun append(value: Int): Vector = append(value.toDouble())

    /**
     * Returns a MutableVector with the same shape and values as this vector.
     * @return MutableVector copy of this vector
     */
    open fun toMutableVector(): MutableVector {
        if (dimensions == 1)
            return MutableVector(shape[0]) { scalarValues[it] }

        val mutableVector = MutableVector(*shape) { 0.0 }
        (0 until shape[0]).forEach {
            mutableVector.vectorValues[it] = vectorValues[it].toMutableVector()
        }
        return mutableVector
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
     * Returns true if the other vector has the same shape and approximately
     * the same values as this vector. Approximate equality can be useful in
     * cases where floating point arithmetic makes calculated values
     * slightly inaccurate.
     * @param other vector to check equality against
     * @return true if the vectors have approximately equivalent values, false otherwise
     */
    fun approxEquals(other: Vector, tolerance: Double = 0.00000001): Boolean =
        shapeEquals(other.shape) &&
            (0 until scalarValues.size).all {
                abs(scalarValues[it] - other.scalarValues[it]) < tolerance
            } &&
            (0 until vectorValues.size).all {
                vectorValues[it].approxEquals(other.vectorValues[it], tolerance)
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
        if (dimensions > 2 || (dimensions == 1 && shape[0] != 1) || (dimensions == 2 && shape[0] != shape[1])) {
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
