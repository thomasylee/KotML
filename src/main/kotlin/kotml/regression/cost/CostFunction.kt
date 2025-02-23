package kotml.regression.cost

import kotml.math.Vector
import kotml.regression.cost.loss.LossFunction

abstract class CostFunction(val lossFunction: LossFunction) {
    /**
     * Returns the evaluated value of the cost function.
     * @param estimates row vector of estimates
     * @param targets row vector of target values
     * @return aggregated cost of the errors
     */
    abstract fun evaluate(estimates: Vector, targets: Vector): Double

    /**
     * Returns the gradient of the cost for each estimate.
     * @param estimates row vector of estimates
     * @param targets row vector of target values
     * @return gradient of the cost for each estimate
     */
    abstract fun gradient(estimates: Vector, targets: Vector): Vector
}
