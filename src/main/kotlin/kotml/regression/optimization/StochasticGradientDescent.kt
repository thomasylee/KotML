package kotml.regression.optimization

import kotml.extensions.* // ktlint-disable no-wildcard-imports
import kotml.math.Vector
import kotml.regression.functions.FunctionModel
import kotml.regression.objectives.CostFunction

/**
 * StochasticGradientDescent incrementally estimates weights based on
 * observations as they are provided.
 */
class StochasticGradientDescent(
    val stepSize: Double,
    regressorCount: Int,
    function: FunctionModel,
    val costFunction: CostFunction,
    includeBias: Boolean,
    weights: DoubleArray = DoubleArray(
        regressorCount + if (includeBias) 1 else 0
    )
) : WeightedOptimizer(regressorCount, function, costFunction, includeBias, weights) {
    constructor(
        stepSize: Double,
        regressorCount: Int,
        function: FunctionModel,
        costFunction: CostFunction,
        weights: DoubleArray = DoubleArray(regressorCount + 1)
    ) : this(stepSize, regressorCount, function, costFunction, weights.size > regressorCount, weights)

    internal override fun addObservation(response: Double, regressors: Vector) {
        val gradient = costFunction.gradient(function, weights, regressors, response)
        (0 until weights.size).forEach { index ->
            weights[index] = weights[index] - stepSize * gradient(index)
        }
    }
}
