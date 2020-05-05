package kotml.regression

import kotml.distributions.DistributionSampler
import kotml.distributions.UniformSampler

class Weights(
    bias: Double?,
    val coeffs: DoubleArray
) {
    val hasBias: Boolean
    var bias: Double

    init {
        if (bias != null) {
            hasBias = true
            this.bias = bias
        } else {
            hasBias = false
            this.bias = 0.0
        }
    }

    constructor(coeffs: DoubleArray) : this(null, coeffs)

    /**
     * Creates weights with the values determined by the provided sampler.
     * Note that if hasBias is true, the first sample will go to the bias.
     * @param hasBias whether a bias should be included in the weights
     * @param coeffCount number of coefficients to create
     * @param sampler sampler used to set initial values for bias and coefficients
     */
    constructor(
        coeffCount: Int,
        hasBias: Boolean = false,
        sampler: DistributionSampler = UniformSampler(0.0)
    ) : this(
        bias = if (hasBias) sampler.sample() else null,
        coeffs = DoubleArray(coeffCount) { sampler.sample() }
    )

    override fun equals(other: Any?): Boolean =
        other != null && other is Weights && other.bias == bias && (0 until coeffs.size).all { index ->
            coeffs[index] == other.coeffs[index]
        }

    override fun toString(): String =
        "Weights(bias = $bias, coeffs = [${coeffs.joinToString(", ")}])"
}
