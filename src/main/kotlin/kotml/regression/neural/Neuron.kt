package kotml.regression.neural

import kotml.distributions.DistributionSampler
import kotml.distributions.NormalSampler
import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.functions.FunctionModel
import kotml.regression.functions.aggregation.AggregationFunction
import kotml.regression.functions.aggregation.DotProduct

/**
 * `Neuron` represents a single neuron in a neural network.
 */
class Neuron(
    val activationFunction: FunctionModel,
    val weights: Weights,
    val aggregationFunction: AggregationFunction = DotProduct
) {
    constructor(
        activationFunction: FunctionModel,
        regressorCount: Int,
        includeConstant: Boolean = true,
        sampler: DistributionSampler = NormalSampler(),
        aggregationFunction: AggregationFunction = DotProduct
    ) : this(
        activationFunction = activationFunction,
        weights = Weights(regressorCount, includeConstant, sampler),
        aggregationFunction = aggregationFunction
    )

    /**
     * Evaluates a vector of inputs and returns the result.
     * @param regressors input values
     * @return output value
     */
    fun evaluate(regressors: Vector): Double =
        activationFunction.evaluate(
            aggregationFunction.aggregate(weights, regressors)
        )

    /**
     * Returns a copy of the neuron.
     * @return copy of the neuron
     */
    fun copy(): Neuron = Neuron(
        activationFunction = activationFunction,
        weights = weights.copy(),
        aggregationFunction = aggregationFunction
    )

    /**
     * Returns true if `other` is an equivalent `Neuron`.
     * @param other nullable object to compare to this one
     * @return true if other is an equivalent Neuron, or false otherwise
     */
    override fun equals(other: Any?): Boolean =
        other is Neuron && activationFunction == other.activationFunction &&
            weights == other.weights &&
            aggregationFunction == other.aggregationFunction
}
