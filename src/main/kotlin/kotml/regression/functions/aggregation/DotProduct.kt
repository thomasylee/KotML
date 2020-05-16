package kotml.regression.functions.aggregation

import kotml.math.Vector
import kotml.regression.Weights

object DotProduct : AggregationFunction {
    override fun aggregate(weights: Weights, regressors: Vector): Double =
        weights.constant + (weights.coeffs * regressors).sum()[0]

    override fun weightsGradient(weights: Weights, regressors: Vector): Weights =
        if (weights.hasConstant)
            Weights(1.0, regressors)
        else
            Weights(regressors)

    override fun regressorsGradient(weights: Weights, regressors: Vector): Vector =
        weights.coeffs
}
