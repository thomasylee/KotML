package kotml.regression

import kotml.distributions.DistributionSampler
import kotml.distributions.UniformSampler

class Weights(
    val hasBias: Boolean = true,
    setBias: Double = 0.0,
    val coeffs: DoubleArray
) {
    var bias: Double

    init {
        bias = if (hasBias) setBias else 0.0
    }

    /**
     * Creates weights with the values determined by the provided sampler.
     * Note that if hasBias is true, the first sample will go to the bias.
     * @param hasBias whether a bias should be included in the weights
     * @param coeffCount number of coefficients to create
     * @param sampler sampler used to set initial values for bias and coefficients
     */
    constructor(
        hasBias: Boolean = true,
        coeffCount: Int,
        sampler: DistributionSampler = UniformSampler(0.0)
    ) : this(hasBias, sampler.sample(), DoubleArray(coeffCount) {
        sampler.sample()
    })

    constructor(setBias: Double, coeffs: DoubleArray) : this(true, setBias, coeffs)

    constructor(hasBias: Boolean = true, values: DoubleArray) : this(
        hasBias = hasBias,
        setBias = if (hasBias) values[0] else 0.0,
        coeffs =
            if (hasBias)
                DoubleArray(values.size - 1) { values[it - 1] }
            else values
    )

    override fun equals(other: Any?): Boolean =
        other != null && other is Weights && other.bias == bias && (0 until coeffs.size).all { index ->
            coeffs[index] == other.coeffs[index]
        }

    override fun toString(): String =
        "Weights(bias = $bias, coeffs = [${coeffs.joinToString(", ")}])"
}
