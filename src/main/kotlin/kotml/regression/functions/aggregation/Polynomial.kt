package kotml.regression.functions.aggregation

import kotlin.math.pow
import kotml.math.MutableVector
import kotml.math.Vector
import kotml.regression.RegressionException
import kotml.regression.Weights

class Polynomial(val exponents: Vector) : AggregationFunction {
    private val regressorCount = exponents.shape[0]

    init {
        if (exponents.dimensions != 1) {
            throw RegressionException("Exponents must be a row vector")
        }
    }

    constructor(vararg exponents: Double) : this(Vector(*exponents))

    constructor(vararg exponents: Int) : this(Vector(*exponents))

    override fun aggregate(weights: Weights, regressors: Vector): Double {
        validateRegressorsShape(regressors)

        return weights.coeffs.foldIndexed(weights.constant) { index, acc, coeff ->
            acc + coeff * regressors[index].pow(exponents[index])
        }[0]
    }

    override fun weightsGradient(weights: Weights, regressors: Vector): Weights {
        validateRegressorsShape(regressors)

        val coeffGradient = MutableVector(weights.coeffs.shape[0]) { index ->
            regressors[index].pow(exponents[index])
        }
        val constant = if (weights.hasConstant) 1.0 else null
        return Weights(constant, coeffGradient)
    }

    override fun regressorsGradient(weights: Weights, regressors: Vector): Vector {
        validateRegressorsShape(regressors)

        return regressors.mapIndexed { index, value ->
            exponents[index] * weights.coeffs[index] * value.pow(exponents[index] - 1)
        }
    }

    private fun validateRegressorsShape(regressors: Vector) {
        if (regressors.dimensions != 1 || regressors.shape[0] != regressorCount) {
            throw RegressionException(
                "Regressors vector shape ${Vector.shapeToString(regressors.shape)} does not match number of exponents $regressorCount"
            )
        }
    }
}
