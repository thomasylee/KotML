package kotml.regression.functions

import kotml.math.Vector
import kotml.regression.Weights

/**
 * `IdentityFunction` returns a dependent variable value that is equal to
 * the net independent variable value.
 */
object IdentityFunction : FunctionOfLinearRegressors {
    override fun evaluateNetInput(netInput: Double): Double = netInput

    override fun netInputGradient(netInput: Double): Double = 1.0

    override fun weightsGradient(weights: Weights, regressors: Vector): Weights =
        if (weights.hasConstant)
            Weights(1.0, regressors)
        else
            Weights(regressors)
}
