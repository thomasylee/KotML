package kotml.math

class MutableVector : Vector {
    constructor(vararg shape: Int, mapValues: (Int) -> Double) : super(
        shape = *shape,
        mapValues = mapValues
    )

    constructor(vararg values: Double) : super(*values)

    constructor(vararg values: Int) : super(*values)

    constructor(vararg values: MutableVector) : super(*values)

    constructor(vararg values: Vector) : this(
        shape = *Vector.addDimensionToShape(
            values.size,
            values.firstOrNull()?.shape ?: intArrayOf()),
        mapValues = { 0.0 }
    ) {
        values.forEachIndexed { index, value ->
            vectorValues[index] = value.toMutableVector()
        }
    }

    companion object {
        @JvmStatic
        fun zeros(vararg shape: Int): MutableVector = MutableVector(
            shape = *shape,
            mapValues = { 0.0 }
        )

        @JvmStatic
        fun ofVectors(numVectors: Int, mapVectors: (Int) -> MutableVector): MutableVector =
            MutableVector(*Array<MutableVector>(numVectors) { mapVectors(it) })
    }

    /**
     * Returns a new MutableVector whose values are equal to this vector's
     * values. Changes made to the copy do not affect values in the original,
     * and vice versa.
     * @return a copy of this vector
     */
    fun copy(): MutableVector {
        val values = toDoubleArray()
        val vector = MutableVector(*shape) { index ->
            values[index]
        }
        vector.flattenedValues = flattenedValues
        return vector
    }

    operator fun set(vararg indices: Int, value: Int) = set(
        indices = *indices,
        value = value.toDouble()
    )

    operator fun set(vararg indices: Int, value: Double) {
        if (indices.size != dimensions) {
            throw ShapeException(
                "Number of indices [${indices.joinToString(", ")}] does not match vector dimensions $dimensions"
            )
        }
        var vector: Vector = this
        (0 until (indices.size - 1)).forEach {
            vector = vector.vectorValues[indices[it]]
        }
        vector.scalarValues[indices.last()] = value
        flattenedValues = null
    }

    operator fun set(vararg indices: Int, value: Vector) {
        if (indices.size >= dimensions) {
            throw ShapeException(
                "Number of indices [${indices.joinToString(", ")}] does not match vector dimensions $dimensions"
            )
        }
        var vectorShape = shape
        indices.forEach {
            vectorShape = Vector.removeDimensionFromShape(vectorShape)
        }
        if (!value.shapeEquals(vectorShape)) {
            throw ShapeException(
                "Vector shape ${Vector.shapeToString(value.shape)} does not match expected shape ${Vector.shapeToString(vectorShape)}"
            )
        }
        var vector: Vector = this
        (0 until (indices.size - 1)).forEach {
            vector = vector.vectorValues[indices[it]]
        }
        vector.vectorValues[indices.last()] = value
        flattenedValues = null
    }

    /**
     * Adds the value to all elements in the vector.
     * @param value value to add
     */
    operator fun plusAssign(value: Double) {
        if (dimensions == 1)
            (0 until scalarValues.size).forEach { scalarValues[it] += value }
        else
            vectorValues.forEach { vector ->
                vector.toMutableVector().plusAssign(value)
            }
        flattenedValues = null
    }

    /**
     * Adds the value to all elements in the vector.
     * @param value value to add
     */
    operator fun plusAssign(value: Int) = plusAssign(value.toDouble())

    /**
     * Subtracts the value from all elements in the vector.
     * @param value value to subtract
     */
    operator fun minusAssign(value: Double) {
        if (dimensions == 1)
            (0 until scalarValues.size).forEach { scalarValues[it] -= value }
        else
            vectorValues.forEach { vector ->
                vector.toMutableVector().minusAssign(value)
            }
        flattenedValues = null
    }

    /**
     * Subtracts the value from all elements in the vector.
     * @param value value to subtract
     */
    operator fun minusAssign(value: Int) = minusAssign(value.toDouble())

    /**
     * Multiplies the value to all elements in the vector.
     * @param value value to multiply
     */
    operator fun timesAssign(value: Double) {
        if (dimensions == 1)
            (0 until scalarValues.size).forEach { scalarValues[it] *= value }
        else
            vectorValues.forEach { vector ->
                vector.toMutableVector().timesAssign(value)
            }
        flattenedValues = null
    }

    /**
     * Multiplies the value to all elements in the vector.
     * @param value value to multiply
     */
    operator fun timesAssign(value: Int) = timesAssign(value.toDouble())

    /**
     * Divides the value from all elements in the vector.
     * @param value value to divide
     */
    operator fun divAssign(value: Double) {
        if (dimensions == 1)
            (0 until scalarValues.size).forEach { scalarValues[it] /= value }
        else
            vectorValues.forEach { vector ->
                vector.toMutableVector().divAssign(value)
            }
        flattenedValues = null
    }

    /**
     * Divides the value from all elements in the vector.
     * @param value value to divide
     */
    operator fun divAssign(value: Int) = divAssign(value.toDouble())

    override fun toMutableVector(): MutableVector = this
}
