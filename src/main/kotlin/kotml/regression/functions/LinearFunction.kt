package kotml.regression.functions

import kotml.math.Vector
import kotml.regression.Weights

/**
 * `LinearFunction` represents any function that is linear with respect
 * to its regressors.
 */
object LinearFunction : FunctionOfLinearRegressors {
    override fun evaluateNetInput(netInput: Double): Double = netInput

    override fun netInputGradient(netInput: Double): Double = 1.0

    override fun weightsGradient(weights: Weights, regressors: Vector): Weights =
        if (weights.hasConstant)
            Weights(1.0, regressors)
        else
            Weights(regressors)
}
