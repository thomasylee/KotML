package kotml.regression.cost

import kotml.math.Vector
import kotml.regression.cost.loss.LossFunction

class SumCost(lossFunction: LossFunction) : CostFunction(lossFunction) {
    override fun evaluate(estimates: Vector, responses: Vector): Double =
        estimates.foldIndexed(0.0) { index, acc, estimate ->
            acc + lossFunction.evaluate(estimate, responses[index])
        }[0]

    override fun gradient(estimates: Vector, responses: Vector): Double =
        estimates.foldIndexed(0.0) { index, acc, estimate ->
            acc + lossFunction.gradient(estimate, responses[index])
        }[0]
}
