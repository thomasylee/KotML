package kotml.regression

import kotml.math.Matrix
import kotml.math.ShapeException

/**
 * A RegressionEstimator develops a model of any kind of linear function
 * and uses that model to predict the future responses of regressors via
 * the estimate() method. Note that for some estimators, the calculate()
 * method must be called before estimate() can be used, due to batched
 * calculations required to build the model.
 */
abstract class RegressionEstimator(val regressorFunctions: Array<out (Double) -> Double>) {
    val regressorCount = regressorFunctions.size

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
    fun addObservation(response: Double, regressors: Matrix) {
        validateRegressorsShape(regressors)
        addObservationSafe(response, regressors)
    }

    internal abstract fun addObservationSafe(response: Double, regressors: Matrix)

    /**
     * Returns a matrix representation of the model, which is generally a
     * series of weights.
     * @return matrix containing the model representation
     */
    abstract fun calculate(): Matrix

    /**
     * Returns an estimate of the response given a complete set of
     * regressor values.
     * @param regressors values for the independent variables
     * @return response estimated by the model
     */
    fun estimate(regressors: Matrix): Double {
        validateRegressorsShape(regressors)
        return estimateSafe(regressors)
    }

    internal abstract fun estimateSafe(regressors: Matrix): Double

    internal fun validateRegressorsShape(regressors: Matrix) {
        if (regressors.dimensions != 1 || regressors.shape[0] != regressorCount) {
            throw RegressionException(
                "Shape of regressors [${regressors.shape.joinToString(", ")}] must be [$regressorCount]"
            )
        }
    }
}
