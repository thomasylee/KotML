package kotml.regression.optimization

import kotml.distributions.DistributionSampler
import kotml.math.Vector
import kotml.regression.RegressionException
import kotml.regression.Weights
import kotml.regression.functions.FunctionModel
import kotml.regression.objectives.ObjectiveFunction

/**
 * An Optimizer develops a model of any kind of linear function by optimizing
 * an objective function.
 */
abstract class WeightedOptimizer(
    val function: FunctionModel,
    val objectiveFunction: ObjectiveFunction,
    val weights: Weights
) {
    /**
     * Copies all property-like attributes of the optimizer except for
     * the weights. This simplifies specifying an optimizer one time and
     * being able to use it in multiple places (such as for different
     * neurons in a neural network).
     * @param sampler sampler used to initialize the weights
     * @return a copy of this WeightedOptimizer, but with different weights
     */
    internal abstract fun copy(sampler: DistributionSampler): WeightedOptimizer

    /**
     * Adds an observation to the training model.
     * @param response the dependent variable value
     * @param regressors the independent variables
     */
    fun observe(regressors: Vector, response: Double) {
        validateRegressorsShape(regressors)
        addObservation(response, regressors)
    }

    internal abstract fun addObservation(response: Double, regressors: Vector)

    internal fun validateRegressorsShape(regressors: Vector) {
        if (regressors.dimensions != 1 || regressors.shape[0] != weights.coeffs.size) {
            throw RegressionException(
                "Shape of regressors [${regressors.shape.joinToString(", ")}] must be [${weights.coeffs.size}]"
            )
        }
    }
}
