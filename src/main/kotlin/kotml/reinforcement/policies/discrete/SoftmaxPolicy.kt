package kotml.reinforcement.policies.discrete

import kotlin.math.exp
import kotlin.random.Random
import kotml.math.Vector

class SoftmaxPolicy(
    var tau: Double = 1.0,
    random: Random = Random
) : DiscreteBehaviorPolicy(random) {
    var lastQValues: Vector = Vector.zeros(1)
    var distribution: Vector = Vector(1) { 1.0 }

    override fun actionProbabilities(qValuesForState: Vector): Vector {
        if (qValuesForState != lastQValues) {
            lastQValues = qValuesForState
            val maxQ = qValuesForState.max()[0]
            val sum = qValuesForState.fold(0.0) { acc, qValue ->
                acc + exp((qValue - maxQ) / tau)
            }[0]
            distribution = Vector(qValuesForState.shape[0]) { index ->
                exp((qValuesForState[index] - maxQ) / tau) / sum
            }
        }

        return distribution
    }
}
