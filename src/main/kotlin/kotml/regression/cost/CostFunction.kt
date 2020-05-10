package kotml.regression.cost

import kotml.math.Vector
import kotml.regression.cost.loss.LossFunction

abstract class CostFunction(val lossFunction: LossFunction) {
    /**
     * Returns the evaluated value of the cost function.
     * @param estimates row vector of estimates
     * @param responses row vector of target values
     * @return aggregated cost of the errors
     */
    abstract fun evaluate(estimates: Vector, responses: Vector): Double

    /**
     * Returns the aggregated gradient of the cost.
     * @param estimates row vector of estimates
     * @param responses row vector of target values
     * @return gradient of the cost
     */
    abstract fun gradient(estimates: Vector, responses: Vector): Double
}
