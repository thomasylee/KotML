package kotml.regression.functions

import kotlin.math.max
import kotml.math.Vector
import kotml.regression.Weights

/**
 * ReLU is represented by f(x) = max(0, x).
 */
object ReLU : FunctionOfLinearRegressors {
    override fun evaluate(weights: Weights, regressors: Vector): Double =
        max(0.0, (0 until regressors.shape[0]).fold(weights.bias) { sumAcc, index ->
            sumAcc + weights.coeffs[index] * regressors(index)
        })

    override fun netInputGradient(netInput: Double): Double =
        if (netInput < 0.0)
            0.0
        else
            1.0

    override fun weightsGradient(weights: Weights, regressors: Vector): Weights {
        val value = evaluate(weights, regressors)

        // Derivative for all weights is 0 when ReLU == 0.
        if (value == 0.0)
            return Weights(weights.coeffs.size, weights.hasBias)

        val bias = if (weights.hasBias) 1.0 else null
        val coeffs = DoubleArray(regressors.shape[0]) {
            regressors(it)
        }
        return Weights(bias, coeffs)
    }
}
