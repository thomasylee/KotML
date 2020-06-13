package kotml.regression.functions

import kotml.math.Vector

interface FunctionModel {
    /**
     * Returns the function evaluated with the provided input.
     * @param input input value
     * @return function value for the input
     */
    abstract fun evaluate(input: Double): Double

    fun evaluate(input: Vector): Vector = input.map { evaluate(it) }

    /**
     * Returns the gradient of the function with respect to the input.
     * @param input input value
     * @return derivative with respect to the input
     */
    abstract fun derivative(input: Double): Double

    fun derivative(input: Vector): Vector = input.map { derivative(it) }
}
