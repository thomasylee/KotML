package kotml.regression.functions

/**
 * FunctionOfLinearRegressors indicates that a function is linear with
 * respect to its independent variables.
 */
interface FunctionOfLinearRegressors : FunctionModel {
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
}
