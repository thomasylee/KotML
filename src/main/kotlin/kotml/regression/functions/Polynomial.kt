package kotml.regression.functions

import kotlin.math.pow
import kotml.math.Vector
import kotml.regression.RegressionException
import kotml.regression.Weights

class Polynomial(val exponents: Vector) : FunctionModel {
    private val regressorCount = exponents.shape[0]

    init {
        if (exponents.dimensions != 1) {
            throw RegressionException("Exponents must be a row vector")
        }
    }

    constructor(vararg exponents: Double) : this(Vector(*exponents))

    constructor(vararg exponents: Int) : this(Vector(*DoubleArray(exponents.size) { exponents[it].toDouble() }))

    override fun evaluate(weights: Weights, regressors: Vector): Double {
        validateRegressorsShape(regressors)

        return weights.coeffs.foldIndexed(weights.bias) { index, acc, coeff ->
            acc + coeff * regressors(index).pow(exponents(index))
        }
    }

    override fun gradient(weights: Weights, regressors: Vector): Weights {
        validateRegressorsShape(regressors)

        val coeffGradient = DoubleArray(weights.coeffs.size) { index ->
            regressors(index).pow(exponents(index))
        }
        val bias = if (weights.hasBias) 1.0 else null
        return Weights(bias, coeffGradient)
    }

    private fun validateRegressorsShape(regressors: Vector) {
        if (regressors.dimensions != 1 || regressors.shape[0] != regressorCount) {
            throw RegressionException(
                "Regressors vector shape ${Vector.shapeToString(regressors.shape)} does not match number of exponents $regressorCount"
            )
        }
    }
}
