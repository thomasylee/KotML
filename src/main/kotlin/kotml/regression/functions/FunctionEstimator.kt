package kotml.regression.functions

import kotml.math.Vector

interface FunctionEstimator {
    abstract fun estimate(weights: DoubleArray, regressors: Vector): Double

    abstract fun gradient(weights: DoubleArray, regressors: Vector): Vector
}
