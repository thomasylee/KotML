package kotml.regression.optimization

import kotml.math.Vector
import kotml.regression.RegressionException
import kotml.regression.Weights
import kotml.regression.functions.FunctionModel

/**
 * Optimizer optimizes weights for a particular function to minimize a
 * loss or cost.
 */
abstract class Optimizer(
    val function: FunctionModel,
    val weights: Weights
) {
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
        if (regressors.dimensions != 1 || regressors.shape[0] != weights.coeffs.shape[0]) {
            throw RegressionException(
                "Shape of regressors [${regressors.shape.joinToString(", ")}] must be [${weights.coeffs.shape[0]}]"
            )
        }
    }
}
