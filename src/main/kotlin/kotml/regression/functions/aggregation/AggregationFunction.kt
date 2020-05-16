package kotml.regression.functions.aggregation

import kotml.math.Vector
import kotml.regression.Weights

interface AggregationFunction {
    abstract fun aggregate(weights: Weights, regressors: Vector): Double

    abstract fun weightsGradient(weights: Weights, regressors: Vector): Weights

    abstract fun regressorsGradient(weights: Weights, regressors: Vector): Vector
}
