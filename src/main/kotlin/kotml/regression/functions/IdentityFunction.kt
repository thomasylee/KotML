package kotml.regression.functions

import kotml.math.Vector

/**
 * `IdentityFunction` returns a dependent variable value that is equal to
 * the independent variable value.
 */
object IdentityFunction : FunctionModel {
    override fun evaluate(input: Double): Double = input

    override fun evaluate(input: Vector): Vector = input

    override fun derivative(input: Double): Double = 1.0

    override fun derivative(input: Vector): Vector = Vector(*input.shape) { 1.0 }
}
