package kotml.regression.functions

/**
 * PReLU (Parameterized ReLU) is parameter * x when x is < 0 and x when x
 * is >= 0.
 */
open class PReLU(val parameter: Double) : FunctionModel {
    override fun evaluate(input: Double): Double =
        if (input < 0 && parameter == 0.0)
            0.0
        else if (input < 0)
            parameter * input
        else
            input

    override fun derivative(input: Double): Double =
        if (input < 0.0)
            parameter
        else
            1.0
}
