package kotml.regression.optimization

import kotml.extensions.* // ktlint-disable no-wildcard-imports
import kotml.math.Vector
import kotml.regression.RegressionEstimator
import kotml.regression.RegressionException
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
    initWeights: DoubleArray = DoubleArray(regressorCount + 1)
) : RegressionEstimator(regressorCount, function) {
    val weights: DoubleArray

    init {
        if (initWeights.size != regressorCount + 1) {
            throw RegressionException("Number of initial weights must equal the regressorCount + 1")
        }
        weights = initWeights
    }

    internal override fun addObservationSafe(response: Double, regressors: Vector) {
        val gradient = costFunction.gradient(function, weights, regressors, response)
        (0..regressorCount).forEach { index ->
            weights[index] = weights[index] - stepSize * gradient(index)
        }
    }

    /**
     * Returns the weights. Since the weights are estimated as each
     * observation is added, no heavy calculations are required on calls
     * to this method.
     * @return weights
     */
    override fun calculate(): DoubleArray = weights

    internal override fun estimateSafe(regressors: Vector): Double =
        function.evaluate(weights, regressors)
}
