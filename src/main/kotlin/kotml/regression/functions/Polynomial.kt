package kotml.regression.functions

import kotlin.math.pow
import kotml.math.Vector
import kotml.regression.RegressionException

class Polynomial(val exponents: Vector) : FunctionModel {
    private val regressorCount = exponents.shape[0]

    init {
        if (exponents.dimensions != 1) {
            throw RegressionException("Exponents must be a row vector")
        }
    }

    constructor(vararg exponents: Double) : this(Vector(*exponents))

    constructor(vararg exponents: Int) : this(Vector(*DoubleArray(exponents.size) { exponents[it].toDouble() }))

    override fun evaluate(weights: DoubleArray, regressors: Vector): Double {
        validateRegressorsShape(regressors)

        // Offset the regressors if weights[0] is a bias.
        if (weights.size == regressorCount + 1) {
            return (0 until regressorCount).fold(weights[0]) { sumAcc, index ->
                sumAcc + weights[index + 1] * regressors(index).pow(exponents(index))
            }
        }
        return (0 until regressorCount).fold(0.0) { sumAcc, index ->
            sumAcc + weights[index] * regressors(index).pow(exponents(index))
        }
    }

    override fun gradient(weights: DoubleArray, regressors: Vector): Vector {
        validateRegressorsShape(regressors)

        // Offset the regressors if weights[0] is a bias.
        if (weights.size == regressorCount + 1) {
            return Vector(*DoubleArray(weights.size) { index ->
                if (index == 0)
                    1.0
                else
                    regressors(index - 1).pow(exponents(index - 1))
            })
        }
        return Vector(*DoubleArray(weights.size) { index ->
            regressors(index).pow(exponents(index))
        })
    }

    private fun validateRegressorsShape(regressors: Vector) {
        if (regressors.dimensions != 1 || regressors.shape[0] != regressorCount) {
            throw RegressionException(
                "Regressors vector shape ${Vector.shapeToString(regressors.shape)} does not match number of exponents $regressorCount"
            )
        }
    }
}
