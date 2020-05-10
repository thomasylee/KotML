package kotml.regression.cost

import kotml.math.Vector
import kotml.regression.cost.loss.LossFunction

class MeanCost(lossFunction: LossFunction) : CostFunction(lossFunction) {
    private val sumCost = SumCost(lossFunction)

    override fun evaluate(estimates: Vector, targets: Vector): Double =
        sumCost.evaluate(estimates, targets) / estimates.shape[0]

    override fun gradient(estimates: Vector, targets: Vector): Double =
        sumCost.gradient(estimates, targets) / estimates.shape[0]
}
