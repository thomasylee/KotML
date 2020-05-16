package kotml.regression.functions

interface FunctionModel {
    /**
     * Returns the function evaluated with the provided input.
     * @param input input value
     * @return function value for the input
     */
    abstract fun evaluate(input: Double): Double

    /**
     * Returns the gradient of the function with respect to the input.
     * @param input input value
     * @return derivative with respect to the input
     */
    abstract fun derivative(input: Double): Double
}
