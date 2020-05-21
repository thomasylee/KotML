package kotml.regression.optimization

import kotml.extensions.* // ktlint-disable no-wildcard-imports
import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.cost.loss.LossFunction
import kotml.regression.functions.FunctionModel
import kotml.regression.functions.aggregation.AggregationFunction
import kotml.regression.functions.aggregation.DotProduct

/**
 * StochasticGradientDescent incrementally estimates weights based on
 * observations as they are provided.
 */
class StochasticGradientDescent(
    val stepSize: Double,
    val function: FunctionModel,
    val lossFunction: LossFunction,
    val weights: Weights,
    val aggregationFunction: AggregationFunction = DotProduct
) : IterativeOptimizer<Weights, Double>(weights.coeffs.shape[0], 1, weights) {
    constructor(
        stepSize: Double,
        function: FunctionModel,
        lossFunction: LossFunction,
        regressorCount: Int,
        hasConstant: Boolean = true,
        aggregationFunction: AggregationFunction = DotProduct
    ) : this(
        stepSize = stepSize,
        function = function,
        lossFunction = lossFunction,
        weights = Weights(regressorCount, hasConstant),
        aggregationFunction = aggregationFunction
    )

    protected override fun addObservation(regressors: Vector, targets: Vector) {
        observeAndEvaluate(regressors, targets)
    }

    override fun observeAndEvaluate(regressors: Vector, targets: Vector): Double {
        val estimate = function.evaluate(
            aggregationFunction.aggregate(weights, regressors)
        )
        val dF_dIn = lossFunction.derivative(estimate, targets[0])
        val dIn_dWeight = aggregationFunction.weightsGradient(weights, regressors)

        if (weights.hasConstant)
            weights.constant -= stepSize * dF_dIn * dIn_dWeight.constant

        (0 until regressors.shape[0]).forEach { index ->
            weights.coeffs[index] -= stepSize * dF_dIn * dIn_dWeight.coeffs[index]
        }

        return estimate
    }
}
