package kotml.regression.objectives

import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.functions.FunctionModel

object OrdinaryLeastSquares : CostFunction {
    /**
     * Returns the least squares cost.
     * @param function model function
     * @param weights weights used by the estimating function
     * @param regressors independent variable values
     * @param response dependent variable value
     * @return least squares cost
     */
    override fun evaluate(function: FunctionModel, weights: Weights, regressors: Vector, response: Double): Double {
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
    fun gradient(function: FunctionModel, weights: Weights, regressors: Vector, response: Double): Weights =
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
    override fun gradient(function: FunctionModel, weights: Weights, regressors: Vector, response: Double, estimate: Double?): Weights {
        val estimateValue = estimate ?: function.evaluate(weights, regressors)
        val gradient = function.gradient(weights, regressors)
        val coeffs = DoubleArray(weights.coeffs.size) { index ->
            2.0 * (estimateValue - response) * gradient.coeffs[index]
        }
        if (weights.hasBias)
            return Weights(2.0 * (estimateValue - response) * gradient.bias, coeffs)
        return Weights(false, coeffs)
    }
}
