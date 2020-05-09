package kotml.regression

import kotml.distributions.DistributionSampler
import kotml.distributions.UniformSampler
import kotml.math.MutableVector
import kotml.math.Vector

class Weights(
    bias: Double?,
    coeffs: Vector
) {
    val hasBias: Boolean
    var bias: Double
    val coeffs: MutableVector

    init {
        if (bias != null) {
            hasBias = true
            this.bias = bias
        } else {
            hasBias = false
            this.bias = 0.0
        }
        if (coeffs.dimensions != 1) {
            throw RegressionException("Weights must have coeffs with only one dimension")
        }
        this.coeffs = coeffs.toMutableVector()
    }

    constructor(coeffs: Vector) : this(null, coeffs.toMutableVector())

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
        coeffs = MutableVector(coeffCount) { sampler.sample() }
    )

    override fun equals(other: Any?): Boolean =
        other != null && other is Weights && other.bias == bias && (0 until coeffs.shape[0]).all { index ->
            coeffs[index] == other.coeffs[index]
        }

    override fun toString(): String = "Weights(bias = $bias, coeffs = $coeffs)"
}
