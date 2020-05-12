package kotml.regression.functions

import kotlin.math.exp
import kotlin.math.pow
import kotml.math.MutableVector
import kotml.math.Vector
import kotml.regression.Weights

object LogisticFunction : FunctionOfLinearRegressors {
    override fun evaluateNetInput(netInput: Double): Double =
        1.0 / (1.0 + exp(-netInput))

    override fun netInputGradient(netInput: Double): Double {
        val expInput = exp(netInput)
        return expInput / (expInput + 1).pow(2)
    }

    override fun weightsGradient(weights: Weights, regressors: Vector): Weights {
        val sum = calculateNetInput(weights, regressors)
        val expSum = exp(-sum)
        val coeffs = MutableVector(weights.coeffs.shape[0]) { index ->
            regressors[index] * expSum / (expSum + 1.0).pow(2.0)
        }

        val constant =
            if (weights.hasConstant)
                expSum / (expSum + 1.0).pow(2.0)
            else
                null
        return Weights(constant, coeffs)
    }
}
