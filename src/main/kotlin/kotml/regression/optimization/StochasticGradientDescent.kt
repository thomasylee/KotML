package kotml.regression.optimization

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
        weights = Weights(hasBias, regressorCount)
    )

    internal override fun addObservation(response: Double, regressors: Vector) {
        val gradient = costFunction.gradient(function, weights, regressors, response)

        if (weights.hasBias)
            weights.bias = weights.bias - stepSize * gradient.bias

        weights.coeffs.forEachIndexed { index, coeff ->
            weights.coeffs[index] = coeff - stepSize * gradient.coeffs[index]
        }
    }
}
