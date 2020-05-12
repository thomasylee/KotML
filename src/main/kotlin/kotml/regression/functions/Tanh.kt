package kotml.regression.functions

import kotlin.math.cosh
import kotlin.math.pow
import kotlin.math.tanh
import kotml.extensions.times
import kotml.math.Vector
import kotml.regression.Weights

/**
 * `Tanh` models the hyperbolic tangent function.
 */
object Tanh : FunctionOfLinearRegressors {
    val gradientOfOne = netInputGradient(1.0)

    override fun evaluateNetInput(netInput: Double): Double = tanh(netInput)

    override fun netInputGradient(netInput: Double): Double =
        1.0 / cosh(netInput).pow(2)

    override fun weightsGradient(weights: Weights, regressors: Vector): Weights {
        val dF_dNetIn = netInputGradient(calculateNetInput(weights, regressors))

        // Differential chain rule!
        val coeffs = dF_dNetIn * regressors

        if (weights.hasConstant)
            return Weights(gradientOfOne, coeffs)
        return Weights(coeffs)
    }
}
