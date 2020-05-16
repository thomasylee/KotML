package kotml.regression.functions

import kotlin.math.exp
import kotlin.math.pow

object LogisticFunction : FunctionModel {
    override fun evaluate(input: Double): Double =
        1.0 / (1.0 + exp(-input))

    override fun derivative(input: Double): Double {
        val expInput = exp(input)
        return expInput / (expInput + 1).pow(2)
    }
}
