package kotml.regression.functions.aggregation

import kotlin.math.exp
import kotml.math.Vector
import kotml.regression.Weights

/**
 * `Softmax` evaluates the softmax function for a particular regressor
 * index. For example, if there are 3 regressors (5, 6, 7), and regressorIndex
 * is set to 1, then the softmax value would be exp(6) / (exp(5) + exp(6) +
 * exp(7)). Weights are completely ignored in this aggregation function.
 */
class Softmax(val regressorIndex: Int) : AggregationFunction {
    /**
     * Returns the softmax value of the regressors for the regressorIndex.
     * @param weights ignored parameter
     * @param regressors independent variable values
     * @return softmax value for the regressor at regressorIndex
     */
    override fun aggregate(weights: Weights, regressors: Vector): Double {
        val max = regressors.max()[0]
        val sum = regressors.foldIndexed(0.0) { index, acc, _ ->
            acc + exp(regressors[index] - max)
        }[0]
        return exp(regressors[regressorIndex] - max) / sum
    }

    /**
     * Returns 0 for the weight constant (if applicable) and all coefficients.
     * @param weights ignored parameter
     * @param regressors ignored parameter
     * @return gradient of the weights, which is always 0
     */
    override fun weightsGradient(weights: Weights, regressors: Vector): Weights =
        if (weights.hasConstant)
            Weights(0.0, Vector.zeros(regressors.shape[0]))
        else
            Weights(Vector.zeros(regressors.shape[0]))

    /**
     * Returns the gradient of the softmax regressors.
     * @param weights ignored parameter
     * @param regressors independent variable values
     * @return vector containing the softmax gradient for each regressor
     */
    override fun regressorsGradient(weights: Weights, regressors: Vector): Vector {
        val max = regressors.max()[0]
        val sum = regressors.foldIndexed(0.0) { index, acc, _ ->
            acc + exp(regressors[index] - max)
        }[0]
        return Vector(regressors.shape[0]) { index ->
            val softmax = exp(regressors[index] - max) / sum
            if (index == regressorIndex)
                softmax * (1.0 - softmax)
            else
                -softmax * softmax
        }
    }
}
