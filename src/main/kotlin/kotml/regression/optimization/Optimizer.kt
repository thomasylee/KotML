package kotml.regression.optimization

import kotml.math.Vector
import kotml.regression.RegressionException

/**
 * Optimizer optimizes weights for a function to minimize a loss or cost.
 */
sealed class Optimizer<T>(
    val regressorCount: Int,
    val outputCount: Int,
    val model: T
) {
    /**
     * Adds an observation to the training model.
     * @param regressors the independent variables
     * @param targets the dependent variable values
     */
    fun observe(regressors: Vector, targets: Vector) {
        validateShape("regressors", regressors, regressorCount)
        validateShape("targets", targets, outputCount)
        addObservation(regressors, targets)
    }

    protected abstract fun addObservation(regressors: Vector, targets: Vector)

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
abstract class IterativeOptimizer<T>(
    regressorCount: Int,
    outputCount: Int,
    model: T
) : Optimizer<T>(regressorCount, outputCount, model)

/**
 * BatchOptimizer develops a model of any kind of linear function by
 * updating weights in batches.
 */
abstract class BatchOptimizer<T>(
    regressorCount: Int,
    outputCount: Int,
    model: T
) : Optimizer<T>(regressorCount, outputCount, model) {
    abstract fun processBatch()
}
