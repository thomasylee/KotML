package kotml.regression.optimization

import kotml.extensions.* // ktlint-disable no-wildcard-imports
import kotml.math.Vector
import kotml.regression.RegressionEstimator
import kotml.regression.functions.FunctionEstimator
import kotml.regression.objectives.CostFunction

/**
 * StochasticGradientDescent incrementally estimates weights based on
 * observations as they are provided.
 */
class StochasticGradientDescent(
    val stepSize: Double,
    function: FunctionEstimator,
    val costFunction: CostFunction
) : RegressionEstimator(function) {
    val weights: DoubleArray = DoubleArray(regressorCount + 1)

    internal override fun addObservationSafe(response: Double, regressors: Vector) {
        val gradient = costFunction.gradient(function, weights, regressors, response)
        (0..regressorCount).forEach { index ->
            weights[index] = weights[index] - stepSize * gradient(index)
        }
    }

    /**
     * Returns a Vector instance containing the weights. Since the weights
     * are estimated as each observation is added, no heavy calculations
     * are required on calls to calculate() aside from copying the weight
     * values into the new Vector instance.
     * @return weights contained in a vector
     */
    override fun calculate(): Vector = Vector(*weights)

    internal override fun estimateSafe(regressors: Vector): Double =
        function.estimate(weights, regressors)
}
