package kotml.regression.objectives

import kotml.math.Vector
import kotml.regression.functions.FunctionModel

object OrdinaryLeastSquares : CostFunction() {
    /**
     * Returns the least squares cost.
     * @param function model function
     * @param weights weights used by the estimating function
     * @param regressors independent variable values
     * @param response dependent variable value
     * @return least squares cost
     */
    override fun cost(function: FunctionModel, weights: DoubleArray, regressors: Vector, response: Double): Double {
        val error = function.evaluate(weights, regressors) - response
        return error * error
    }

    /**
     * Returns the gradient of the least squares cost at a given point in
     * the provided estimating function.
     * @param function model function
     * @param weights weights used by the estimating function
     * @param regressors independent variable values
     * @param response dependent variable value
     * @return gradient of the least squares error
     */
    fun gradient(function: FunctionModel, weights: DoubleArray, regressors: Vector, response: Double): Vector =
        gradient(function, weights, regressors, response, null)

    /**
     * Returns the gradient of the least squares cost at a given point in
     * the provided estimating function.
     * @param function model function
     * @param weights weights used by the estimating function
     * @param regressors independent variable values
     * @param response dependent variable value
     * @param estimate optional estimate from the estimating function
     * @return gradient of the least squares error
     */
    override fun gradient(function: FunctionModel, weights: DoubleArray, regressors: Vector, response: Double, estimate: Double?): Vector {
        val estimateValue = estimate ?: function.evaluate(weights, regressors)
        return function.gradient(weights, regressors).map { deriv ->
            2.0 * (estimateValue - response) * deriv
        }
    }
}
