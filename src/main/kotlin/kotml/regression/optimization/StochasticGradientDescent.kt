package kotml.regression.optimization

import kotml.distributions.DistributionSampler
import kotml.distributions.UniformSampler
import kotml.extensions.* // ktlint-disable no-wildcard-imports
import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.functions.FunctionModel
import kotml.regression.objectives.CostFunction

/**
 * StochasticGradientDescent incrementally estimates weights based on
 * observations as they are provided.
 */
class StochasticGradientDescent(
    val stepSize: Double,
    function: FunctionModel,
    val costFunction: CostFunction,
    weights: Weights
) : WeightedOptimizer(function, costFunction, weights) {
    constructor(
        stepSize: Double,
        function: FunctionModel,
        costFunction: CostFunction,
        regressorCount: Int,
        hasBias: Boolean = true
    ) : this(
        stepSize = stepSize,
        function = function,
        costFunction = costFunction,
        weights = Weights(regressorCount, hasBias)
    )

    internal fun copy(): WeightedOptimizer = copy(UniformSampler(0.0))

    internal override fun copy(sampler: DistributionSampler): WeightedOptimizer =
        StochasticGradientDescent(
            stepSize = stepSize,
            function = function,
            costFunction = costFunction,
            weights = Weights(weights.coeffs.size, weights.hasBias, sampler))

    internal override fun addObservation(response: Double, regressors: Vector) {
        val estimate = function.evaluate(weights, regressors)
        val gradient = costFunction.gradient(response, estimate)

        if (weights.hasBias)
            weights.bias -= stepSize * gradient

        weights.coeffs.forEachIndexed { index, _ ->
            weights.coeffs[index] -= stepSize * gradient * regressors(index)
        }
    }
}
