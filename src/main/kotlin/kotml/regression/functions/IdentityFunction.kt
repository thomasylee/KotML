package kotml.regression.functions

/**
 * `IdentityFunction` returns a dependent variable value that is equal to
 * the independent variable value.
 */
object IdentityFunction : FunctionModel {
    override fun evaluate(input: Double): Double = input

    override fun derivative(input: Double): Double = 1.0
}
