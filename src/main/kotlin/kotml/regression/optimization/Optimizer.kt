package kotml.regression.optimization

import kotml.math.Vector
import kotml.regression.RegressionException

/**
 * Optimizer optimizes weights for a function to minimize a loss or cost.
 */
sealed class Optimizer<M>(
    val regressorCount: Int,
    val outputCount: Int,
    val model: M
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
abstract class IterativeOptimizer<M, V>(
    regressorCount: Int,
    outputCount: Int,
    model: M
) : Optimizer<M>(regressorCount, outputCount, model) {
    /**
     * Returns the evaluated output of the optimizer while also optimizing
     * the output and targets.
     * @param regressors independent variable values
     * @param targets target dependent variable values
     * @return evaluated dependent variable values
     */
    abstract fun observeAndEvaluate(regressors: Vector, targets: Vector): V

    /**
     * Returns the evaluated output of the optimizer while also optimizing
     * the output and targets as a single batch.
     * @param regressorsBatch batch of regressors
     * @param targetsBatch batch of targets
     * @return evaluated batches
     */
    abstract fun batchObserveAndEvaluate(regressorsBatch: Vector, targetsBatch: Vector): Vector
}

/**
 * BatchOptimizer develops a model of any kind of linear function by
 * updating weights in batches.
 */
abstract class BatchOptimizer<M>(
    regressorCount: Int,
    outputCount: Int,
    model: M
) : Optimizer<M>(regressorCount, outputCount, model) {
    abstract fun processBatch()
}
