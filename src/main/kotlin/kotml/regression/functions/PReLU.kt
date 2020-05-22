package kotml.regression.functions

/**
 * PReLU (Parameterized ReLU) is parameter * x when x is < 0 and x when x
 * is >= 0.
 */
open class PReLU(val alpha: Double) : FunctionModel {
    override fun evaluate(input: Double): Double =
        if (input < 0 && alpha == 0.0)
            0.0
        else if (input < 0)
            alpha * input
        else
            input

    override fun derivative(input: Double): Double =
        if (input < 0.0)
            alpha
        else
            1.0
}
