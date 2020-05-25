package kotml.regression.functions

import kotlin.math.exp

/**
 * Exponential linear unit (ELU) yields alpha * (exp(x) - 1) when x <= 0,
 * and x when x > 0.
 */
class ELU(val alpha: Double) : FunctionModel {
    override fun evaluate(input: Double): Double =
        if (input < 0)
            alpha * (exp(input) - 1.0)
        else
            input

    override fun derivative(input: Double): Double =
        if (input < 0.0)
            alpha + evaluate(input)
        else
            1.0
}
