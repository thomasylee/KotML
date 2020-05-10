package kotml.regression

import kotml.distributions.DistributionSampler
import kotml.distributions.UniformSampler
import kotml.math.MutableVector
import kotml.math.Vector

class Weights(
    constant: Double?,
    coeffs: Vector
) {
    val hasConstant: Boolean
    var constant: Double
    val coeffs: MutableVector

    init {
        if (constant != null) {
            hasConstant = true
            this.constant = constant
        } else {
            hasConstant = false
            this.constant = 0.0
        }
        if (coeffs.dimensions != 1) {
            throw RegressionException("Weights must have coeffs with only one dimension")
        }
        this.coeffs = coeffs.toMutableVector()
    }

    constructor(coeffs: Vector) : this(null, coeffs.toMutableVector())

    /**
     * Creates weights with the values determined by the provided sampler.
     * Note that if hasBias is true, the first sample will go to the constant.
     * @param hasConstant whether a constant should be included in the weights
     * @param coeffCount number of coefficients to create
     * @param sampler sampler used to set initial values for constant and coefficients
     */
    constructor(
        coeffCount: Int,
        hasConstant: Boolean = false,
        sampler: DistributionSampler = UniformSampler(0.0)
    ) : this(
        constant = if (hasConstant) sampler.sample() else null,
        coeffs = MutableVector(coeffCount) { sampler.sample() }
    )

    fun toVector(): Vector = coeffs.append(constant)

    override fun equals(other: Any?): Boolean =
        other is Weights && other.hasConstant == hasConstant &&
            (!hasConstant || other.constant == constant) && other.coeffs == coeffs

    override fun toString(): String = "Weights(constant = $constant, coeffs = $coeffs)"
}
