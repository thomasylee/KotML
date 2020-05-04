package kotml.regression.functions

import kotml.math.Vector

interface FunctionModel {
    abstract fun evaluate(weights: DoubleArray, regressors: Vector): Double

    abstract fun gradient(weights: DoubleArray, regressors: Vector): Vector
}
