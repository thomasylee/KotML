package kotml.regression.functions

import kotlin.math.max

/**
 * ReLU is represented by f(x) = max(0, x).
 */
object ReLU : FunctionModel {
    override fun evaluate(input: Double): Double = max(0.0, input)

    override fun derivative(input: Double): Double =
        if (input <= 0.0)
            0.0
        else
            1.0
}
