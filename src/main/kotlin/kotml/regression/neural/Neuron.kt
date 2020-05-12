package kotml.regression.neural

import kotml.distributions.DistributionSampler
import kotml.distributions.NormalSampler
import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.functions.FunctionOfLinearRegressors

/**
 * `Neuron` represents a single neuron in a neural network.
 */
class Neuron(
    val activationFunction: FunctionOfLinearRegressors,
    val weights: Weights
) {
    constructor(
        activationFunction: FunctionOfLinearRegressors,
        regressorCount: Int,
        includeConstant: Boolean = true,
        sampler: DistributionSampler = NormalSampler()
    ) : this(
        activationFunction = activationFunction,
        weights = Weights(regressorCount, includeConstant, sampler)
    )

    /**
     * Evaluates a vector of inputs and returns the result.
     * @param regressors input values
     * @return output value
     */
    fun evaluate(regressors: Vector): Double =
        activationFunction.evaluate(weights, regressors)
}
