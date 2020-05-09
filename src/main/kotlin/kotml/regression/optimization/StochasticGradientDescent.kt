package kotml.regression.optimization

import kotml.extensions.* // ktlint-disable no-wildcard-imports
import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.cost.loss.LossFunction
import kotml.regression.functions.FunctionModel

/**
 * StochasticGradientDescent incrementally estimates weights based on
 * observations as they are provided.
 */
class StochasticGradientDescent(
    val stepSize: Double,
    function: FunctionModel,
    lossFunction: LossFunction,
    weights: Weights
) : IterativeOptimizer(function, lossFunction, weights) {
    constructor(
        stepSize: Double,
        function: FunctionModel,
        lossFunction: LossFunction,
        regressorCount: Int,
        hasBias: Boolean = true
    ) : this(
        stepSize = stepSize,
        function = function,
        lossFunction = lossFunction,
        weights = Weights(regressorCount, hasBias)
    )

    internal override fun addObservation(response: Double, regressors: Vector) {
        val estimate = function.evaluate(weights, regressors)
        val gradient = lossFunction.gradient(estimate, response)

        if (weights.hasBias)
            weights.bias -= stepSize * gradient

        (0 until regressors.shape[0]).forEach { index ->
            weights.coeffs[index] -= stepSize * gradient * regressors[index]
        }
    }
}
