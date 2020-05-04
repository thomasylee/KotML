package kotml.regression.functions

import kotlin.math.exp
import kotlin.math.pow
import kotml.math.Vector

object LogisticFunction : FunctionModel {
    override fun evaluate(weights: DoubleArray, regressors: Vector): Double =
        if (weights.size == regressors.shape[0] + 1) {
            1.0 / (1.0 + exp(
                -(0 until regressors.shape[0]).fold(weights[0]) { sumAcc, index ->
                    sumAcc + weights[index + 1] * regressors(index)
                }
            ))
        } else {
            1.0 / (1.0 + exp(
                -(0 until regressors.shape[0]).fold(0.0) { sumAcc, index ->
                    sumAcc + weights[index] * regressors(index)
                }
            ))
        }

    override fun gradient(weights: DoubleArray, regressors: Vector): Vector {
        // Offset the regressors if weights[0] is a bias.
        val hasBias = weights.size == regressors.shape[0] + 1
        val sum =
            if (hasBias) {
                (0 until regressors.shape[0]).fold(weights[0]) { acc, index ->
                    acc + weights[index + 1] * regressors(index)
                }
            } else {
                (0 until regressors.shape[0]).fold(0.0) { acc, index ->
                    acc + weights[index] * regressors(index)
                }
            }
        val expSum = exp(-sum)

        if (hasBias) {
            return Vector(*DoubleArray(weights.size) { index ->
                if (index == 0)
                    expSum / (expSum + 1.0).pow(2.0)
                else
                    regressors(index - 1) * expSum / (expSum + 1.0).pow(2.0)
            })
        }
        return Vector(*DoubleArray(weights.size) { index ->
            regressors(index) * expSum / (expSum + 1.0).pow(2.0)
        })
    }
}
