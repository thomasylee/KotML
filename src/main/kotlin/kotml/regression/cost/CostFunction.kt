package kotml.regression.cost

import kotml.math.Vector
import kotml.regression.functions.FunctionEstimator

abstract class CostFunction {
    abstract fun cost(function: FunctionEstimator, weights: DoubleArray, regressors: Vector, response: Double): Double

    abstract fun gradient(function: FunctionEstimator, weights: DoubleArray, regressors: Vector, response: Double, estimate: Double? = null): Vector
}
