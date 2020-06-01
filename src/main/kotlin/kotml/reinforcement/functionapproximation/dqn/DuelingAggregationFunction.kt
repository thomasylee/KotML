package kotml.reinforcement.functionapproximation.dqn

import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.functions.aggregation.AggregationFunction

/**
 * `DuelingAggregationFunction` adds a state value and advantage values with
 * mean-advantage adjustment as described in equation 9 of "Dueling Network
 * Architectures for Deep Reinforcement Learning" (2015) by Ziyu Wang, et al.
 * The first regressor value should be the state value, and the remaining
 * regressor values should be the advantage values for each action.
 *
 * References:
 * * Dueling Network Architectures for Deep Reinforcement Learning - 2015 -
 *   Ziyu Wang, Tom Schaul, Matteo Hessel, et al.
 */
class DuelingAggregationFunction(val actionIndex: Int) : AggregationFunction {
    /**
     * Returns the mean advantage value of the given regressors, where the
     * first regressor is the state value and the remaining regressors are
     * the advantage values.
     * @param regressors state value and advantage values
     * @return mean advantage value
     */
    fun meanAdvantage(regressors: Vector): Double =
        regressors.foldIndexed(0.0) { index, acc, advantage ->
            if (index == 0)
                acc
            else
                acc + advantage
        }[0] / (regressors.shape[0] - 1.0)

    /**
     * Returns the sum of the state value (regressors[0]) and advantage
     * value (regressors[actionIndex]) minus the mean advantage.
     * @param weights ignored parameter
     * @param regressors independent variable values
     * @return estimated action value
     */
    override fun aggregate(weights: Weights, regressors: Vector): Double =
        regressors[0] + regressors[actionIndex + 1] - meanAdvantage(regressors)

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

    override fun regressorsGradient(weights: Weights, regressors: Vector): Vector =
        Vector(regressors.shape[0]) { index ->
            if (index == 0)
                1.0
            else if (index == actionIndex + 1)
                1 - 1.0 / (regressors.shape[0] - 1.0)
            else
                0.0
        }
}
