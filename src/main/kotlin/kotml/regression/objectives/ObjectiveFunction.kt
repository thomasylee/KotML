package kotml.regression.objectives

import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.functions.FunctionModel

interface ObjectiveFunction {
    abstract fun evaluate(function: FunctionModel, weights: Weights, regressors: Vector, response: Double): Double

    abstract fun gradient(function: FunctionModel, weights: Weights, regressors: Vector, response: Double, estimate: Double? = null): Weights
}
