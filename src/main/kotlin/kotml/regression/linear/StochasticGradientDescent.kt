package kotml.regression.linear

import kotml.math.Matrix
import kotml.regression.RegressionEstimator

/**
 * StochasticGradientDescent incrementally estimates weights based on
 * observations as they are provided.
 */
class StochasticGradientDescent(
    val stepSize: Double = 0.0001,
    vararg regressorFunctions: (Double) -> Double
) : RegressionEstimator(regressorFunctions) {
    val weights: DoubleArray = DoubleArray(regressorCount + 1)

    internal override fun addObservationSafe(response: Double, regressors: Matrix) {
        val newWeights = DoubleArray(regressorCount + 1) { index ->
            weights[index] - stepSize * 2.0 * DoubleArray(index).fold(1.0) { prodAcc, _ ->
                prodAcc * regressors(index - 1)
            } * (0..regressorCount).fold(-response) { sumAcc, term ->
                if (term == 0) {
                    sumAcc + weights[term]
                } else {
                    sumAcc + weights[term] * regressorFunctions[term - 1](regressors(term - 1))
                }
            }
        }
        System.arraycopy(newWeights, 0, weights, 0, regressorCount + 1)
    }

    /**
     * Returns a Matrix instance containing the weights. Since the weights
     * are estimated as each observation is added, no heavy calculations
     * are required on calls to calculate() aside from copying the weight
     * values into the new Matrix instance.
     * @return weights contained in a matrix
     */
    override fun calculate(): Matrix = Matrix(*weights)

    internal override fun estimateSafe(regressors: Matrix): Double = (0..regressorCount).fold(0.0) { acc, index ->
        acc + weights[index] * (0 until index).fold(1.0) { prodAcc, _ ->
            prodAcc * regressors(index - 1)
        }
    }
}
