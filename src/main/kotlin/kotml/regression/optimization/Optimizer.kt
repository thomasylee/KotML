package kotml.regression.optimization

import kotml.math.Vector
import kotml.regression.RegressionException

/**
 * Optimizer optimizes weights for a function to minimize a loss or cost.
 */
sealed class Optimizer(val regressorCount: Int, val outputCount: Int) {
    /**
     * Adds an observation to the training model.
     * @param regressors the independent variables
     * @param response the dependent variable value
     */
    fun observe(regressors: Vector, response: Vector) {
        validateShape("regressors", regressors, regressorCount)
        validateShape("response", response, outputCount)
        addObservation(response, regressors)
    }

    internal abstract fun addObservation(regressors: Vector, response: Vector)

    private fun validateShape(name: String, vector: Vector, count: Int) {
        if (vector.dimensions != 1 || vector.shape[0] != count) {
            throw RegressionException(
                "Shape of $name [${vector.shape.joinToString(", ")}] must be [$count]"
            )
        }
    }
}

/**
 * IterativeOptimizer develops a model of any kind of linear function by
 * iteratively reducing a loss function.
 */
abstract class IterativeOptimizer(
    regressorCount: Int,
    outputCount: Int
) : Optimizer(regressorCount, outputCount)

/**
 * BatchOptimizer develops a model of any kind of linear function by
 * updating weights in batches.
 */
abstract class BatchOptimizer(
    regressorCount: Int,
    outputCount: Int
) : Optimizer(regressorCount, outputCount) {
    abstract fun processBatch()
}
