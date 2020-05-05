package kotml.regression.functions

import kotlin.math.exp
import kotlin.math.pow
import kotml.math.Vector
import kotml.regression.Weights

object LogisticFunction : FunctionModel {
    override fun evaluate(weights: Weights, regressors: Vector): Double =
        1.0 / (1.0 + exp(
            -(0 until regressors.shape[0]).fold(weights.bias) { sumAcc, index ->
                sumAcc + weights.coeffs[index] * regressors(index)
            }
        ))

    override fun gradient(weights: Weights, regressors: Vector): Weights {
        val sum = (0 until regressors.shape[0]).fold(weights.bias) { acc, index ->
            acc + weights.coeffs[index] * regressors(index)
        }
        val expSum = exp(-sum)
        val coeffs = DoubleArray(weights.coeffs.size) { index ->
            regressors(index) * expSum / (expSum + 1.0).pow(2.0)
        }

        val bias =
            if (weights.hasBias)
                expSum / (expSum + 1.0).pow(2.0)
            else
                null
        return Weights(bias, coeffs)
    }
}
