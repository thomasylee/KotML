package kotml.regression.functions

import kotlin.math.exp
import kotlin.math.pow
import kotml.math.Vector

object LogisticFunction : FunctionModel {
    override fun evaluate(weights: DoubleArray, regressors: Vector): Double =
        1.0 / (1.0 + exp(
            -(0 until regressors.shape[0]).fold(weights[0]) { sumAcc, index ->
                sumAcc + weights[index + 1] * regressors(index)
            }
        ))

    override fun gradient(weights: DoubleArray, regressors: Vector): Vector {
        val sum = (0 until regressors.shape[0]).fold(weights[0]) { acc, index ->
            acc + weights[index + 1] * regressors(index)
        }
        val expSum = exp(-sum)

        return Vector(*DoubleArray(weights.size) { index ->
            if (index == 0)
                expSum / (expSum + 1.0).pow(2.0)
            else
                regressors(index - 1) * expSum / (expSum + 1.0).pow(2.0)
            /*
            if (index == 0) {
                val expWeight = exp(-weights[index])
                expWeight / (expWeight + 1.0).pow(2.0)
            } else {
                val regressor = regressors(index - 1)
                val expWeight = exp(-regressor * weights[index])
                regressor * expWeight / (expWeight + 1.0).pow(2.0)
            }
            */
        })
    }
}
