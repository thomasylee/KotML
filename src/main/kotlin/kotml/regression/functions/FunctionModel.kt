package kotml.regression.functions

import kotml.math.Vector
import kotml.regression.Weights

interface FunctionModel {
    abstract fun evaluate(weights: Weights, regressors: Vector): Double

    abstract fun gradient(weights: Weights, regressors: Vector): Weights
}
