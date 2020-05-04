package kotml.regression.objectives

import kotml.math.Vector
import kotml.regression.functions.FunctionModel

abstract class CostFunction {
    abstract fun cost(function: FunctionModel, weights: DoubleArray, regressors: Vector, response: Double): Double

    abstract fun gradient(function: FunctionModel, weights: DoubleArray, regressors: Vector, response: Double, estimate: Double? = null): Vector
}
