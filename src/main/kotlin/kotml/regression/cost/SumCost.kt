package kotml.regression.cost

import kotml.math.Vector
import kotml.regression.cost.loss.LossFunction

class SumCost(lossFunction: LossFunction) : CostFunction(lossFunction) {
    override fun evaluate(estimates: Vector, targets: Vector): Double =
        estimates.foldIndexed(0.0) { index, acc, estimate ->
            acc + lossFunction.evaluate(estimate, targets[index])
        }[0]

    override fun gradient(estimates: Vector, targets: Vector): Vector =
        estimates.mapIndexed { index, estimate ->
            lossFunction.gradient(estimate, targets[index])
        }
}
