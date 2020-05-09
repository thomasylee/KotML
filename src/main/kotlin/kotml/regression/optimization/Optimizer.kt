package kotml.regression.optimization

import kotml.math.Vector
import kotml.regression.RegressionException
import kotml.regression.Weights
import kotml.regression.cost.CostFunction
import kotml.regression.cost.loss.LossFunction
import kotml.regression.functions.FunctionModel

/**
 * Optimizer optimizes weights for a particular function to minimize a
 * loss or cost.
 */
sealed class Optimizer(
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

/**
 * IterativeOptimizer develops a model of any kind of linear function by
 * iteratively reducing a loss function.
 */
abstract class IterativeOptimizer(
    function: FunctionModel,
    val lossFunction: LossFunction,
    weights: Weights
) : Optimizer(function, weights)

/**
 * BatchOptimizer develops a model of any kind of linear function by
 * updating weights in batches.
 */
abstract class BatchOptimizer(
    function: FunctionModel,
    val costFunction: CostFunction,
    weights: Weights
) : Optimizer(function, weights) {
    abstract fun processBatch()
}
