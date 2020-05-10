package kotml.regression.functions

import kotml.math.Vector
import kotml.regression.Weights

/**
 * `LinearFunction` represents any function that is linear with respect
 * to its regressors.
 */
object LinearFunction : FunctionOfLinearRegressors {
    override fun evaluate(weights: Weights, regressors: Vector): Double =
        weights.constant + (weights.coeffs * regressors).sum()[0]

    override fun netInputGradient(netInput: Double): Double = 1.0

    override fun weightsGradient(weights: Weights, regressors: Vector): Weights =
        if (weights.hasConstant)
            Weights(1.0, regressors)
        else
            Weights(regressors)
}
