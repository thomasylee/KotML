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
    }
}
