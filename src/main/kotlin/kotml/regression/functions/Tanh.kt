package kotml.regression.functions

import kotlin.math.cosh
import kotlin.math.pow
import kotlin.math.tanh

/**
 * `Tanh` models the hyperbolic tangent function.
 */
object Tanh : FunctionModel {
    override fun evaluate(input: Double): Double = tanh(input)

    override fun derivative(input: Double): Double =
        1.0 / cosh(input).pow(2)
}
