package kotml.regression.functions

import kotml.math.Vector
import kotml.regression.RegressionException

abstract class FunctionEstimator(val regressorCount: Int) {
    fun estimate(weights: DoubleArray, regressors: Vector): Double {
        validateRegressorsShape(regressors)
        return estimateSafe(weights, regressors)
    }

    protected abstract fun estimateSafe(weights: DoubleArray, regressors: Vector): Double

    fun gradient(weights: DoubleArray, regressors: Vector): Vector {
        validateRegressorsShape(regressors)
        return gradientSafe(weights, regressors)
    }

    protected abstract fun gradientSafe(weights: DoubleArray, regressors: Vector): Vector

    private fun validateRegressorsShape(regressors: Vector) {
        if (regressors.dimensions != 1 || regressors.shape[0] != regressorCount) {
            throw RegressionException(
                "Regressors vector shape ${Vector.shapeToString(regressors.shape)} does not match expected [$regressorCount]"
            )
        }
    }
}
