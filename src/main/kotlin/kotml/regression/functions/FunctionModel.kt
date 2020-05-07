package kotml.regression.functions

import kotml.math.Vector
import kotml.regression.Weights

interface FunctionModel {
    /**
     * Evaluates the function with the given weights and regressors.
     * @param weights weight values
     * @param regressors values of independent variables
     * @return output value of the function
     */
    abstract fun evaluate(weights: Weights, regressors: Vector): Double

    /**
     * Returns the gradient of the function with respect to each weight.
     * @param weights weight values
     * @param regressors values of independent variables
     * @return gradient of the function with respect to each weight
     */
    abstract fun weightsGradient(weights: Weights, regressors: Vector): Weights
}
