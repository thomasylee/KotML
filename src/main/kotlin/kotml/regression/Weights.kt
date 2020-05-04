package kotml.regression

class Weights(
    val hasBias: Boolean = true,
    setBias: Double = 0.0,
    val coeffs: DoubleArray
) {
    var bias: Double

    init {
        bias = if (hasBias) setBias else 0.0
    }

    constructor(hasBias: Boolean = true, coeffCount: Int) : this(hasBias, 0.0, DoubleArray(coeffCount))

    constructor(setBias: Double, coeffs: DoubleArray) : this(true, setBias, coeffs)

    constructor(hasBias: Boolean = true, coeffs: DoubleArray) : this(
        hasBias = hasBias,
        setBias = if (hasBias) coeffs[0] else 0.0,
        coeffs =
            if (hasBias)
                DoubleArray(coeffs.size - 1) { coeffs[it - 1] }
            else coeffs
    )

    override fun equals(other: Any?): Boolean =
        other != null && other is Weights && other.bias == bias && (0 until coeffs.size).all { index ->
            coeffs[index] == other.coeffs[index]
        }

    override fun toString(): String =
        "Weights(bias = $bias, coeffs = [${coeffs.joinToString(", ")}])"
}
