package kotml.regression.functions

import kotml.math.Vector
import kotml.regression.Weights

/**
 * FunctionOfLinearRegressors indicates that a function is linear with
 * respect to its independent variables.
 */
interface FunctionOfLinearRegressors : FunctionModel {
    /**
     * Returns the function evaluated with the provided net input.
     * @param netInput value of the net input
     * @return function value for the net input
     */
    abstract fun evaluateNetInput(netInput: Double): Double

    /**
     * Evaluates the function with net input equal to the sum of the weight
     * coeffs and regressors, plus the weight constant (0 if hasConstant
     * is false).
     * @param weights weights with coefficients and, optionally, a constant
     * @param regressors independent variable values
     * @return function value for the net input
     */
    override fun evaluate(weights: Weights, regressors: Vector): Double =
        evaluateNetInput(calculateNetInput(weights, regressors))

    /**
     * Returns the gradient of the function with respect to the net input
     * For example, if f(x) = x^2, then the inputGradient would return
     * the gradient with respect to x, even though x is a sum of weights
     * times regressors. This gradient simplifies using the differential
     * chain rule, such as when performing backpropagation.
     * @param netInput value of the net input
     * @return derivative with respect to the net input
     */
    abstract fun netInputGradient(netInput: Double): Double

    fun calculateNetInput(weights: Weights, regressors: Vector): Double =
        weights.constant + (weights.coeffs * regressors).sum()[0]
}
