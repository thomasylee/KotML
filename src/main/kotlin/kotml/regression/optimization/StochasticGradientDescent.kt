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
    val function: FunctionModel,
    val lossFunction: LossFunction,
    val weights: Weights
) : IterativeOptimizer<Weights>(weights.coeffs.shape[0], 1, weights) {
    constructor(
        stepSize: Double,
        function: FunctionModel,
        lossFunction: LossFunction,
        regressorCount: Int,
        hasConstant: Boolean = true
    ) : this(
        stepSize = stepSize,
        function = function,
        lossFunction = lossFunction,
        weights = Weights(regressorCount, hasConstant)
    )

    protected override fun addObservation(regressors: Vector, targets: Vector) {
        val estimate = function.evaluate(weights, regressors)
        val gradient = lossFunction.gradient(estimate, targets[0])

        if (weights.hasConstant)
            weights.constant -= stepSize * gradient

        (0 until regressors.shape[0]).forEach { index ->
            weights.coeffs[index] -= stepSize * gradient * regressors[index]
        }
    }
}
