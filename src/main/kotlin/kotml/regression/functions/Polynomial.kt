package kotml.regression.functions

import kotlin.math.pow
import kotml.math.Vector
import kotml.regression.RegressionException

class Polynomial(val exponents: Vector) : FunctionEstimator(exponents.shape[0]) {
    init {
        if (exponents.dimensions != 1) {
            throw RegressionException("Exponents must be a row vector")
        }
    }

    protected override fun estimateSafe(weights: DoubleArray, regressors: Vector): Double =
        (0 until regressorCount).fold(weights[0]) { sumAcc, index ->
            sumAcc + weights[index + 1] * regressors(index).pow(exponents(index))
        }

    protected override fun gradientSafe(weights: DoubleArray, regressors: Vector): Vector =
        Vector(*DoubleArray(weights.size) { index ->
            if (index == 0)
                1.0
            else
                regressors(index - 1).pow(exponents(index - 1))
        })
}
