package kotml.regression

import kotml.math.ShapeException
import kotml.math.Vector
import kotml.regression.functions.FunctionEstimator

/**
 * A RegressionEstimator develops a model of any kind of linear function
 * and uses that model to predict the future responses of regressors via
 * the estimate() method. Note that for some estimators, the calculate()
 * method must be called before estimate() can be used, due to batched
 * calculations required to build the model.
 */
abstract class RegressionEstimator(
    val regressorCount: Int,
    val function: FunctionEstimator
) {
    init {
        if (regressorCount < 1) {
            println(regressorCount)
            throw ShapeException("regressorCount must be at least 1")
        }
    }

    /**
     * Adds an observation to the training model.
     * @param response the dependent variable value
     * @param regressors the independent variables
     */
    fun addObservation(response: Double, regressors: Vector) {
        validateRegressorsShape(regressors)
        addObservationSafe(response, regressors)
    }

    internal abstract fun addObservationSafe(response: Double, regressors: Vector)

    /**
     * Returns a vector representation of the model, which is generally a
     * series of weights.
     * @return vector containing the model representation
     */
    abstract fun calculate(): Vector

    /**
     * Returns an estimate of the response given a complete set of
     * regressor values.
     * @param regressors values for the independent variables
     * @return response estimated by the model
     */
    fun estimate(regressors: Vector): Double {
        validateRegressorsShape(regressors)
        return estimateSafe(regressors)
    }

    internal abstract fun estimateSafe(regressors: Vector): Double

    internal fun validateRegressorsShape(regressors: Vector) {
        if (regressors.dimensions != 1 || regressors.shape[0] != regressorCount) {
            throw RegressionException(
                "Shape of regressors [${regressors.shape.joinToString(", ")}] must be [$regressorCount]"
            )
        }
    }
}
