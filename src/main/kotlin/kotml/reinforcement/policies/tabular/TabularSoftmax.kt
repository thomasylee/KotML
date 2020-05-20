package kotml.reinforcement.policies.tabular

import kotlin.math.exp
import kotlin.random.Random
import kotml.math.Vector

class TabularSoftmax(
    var tau: Double = 0.5,
    val random: Random = Random
) : TabularPolicy {
    var lastQValues: Vector = Vector.zeros(1)
    var distribution: Vector = Vector(1) { 1.0 }

    override fun chooseAction(qValuesForState: Vector): Int {
        val probabilities = actionProbabilities(qValuesForState)
        val randomValue = random.nextDouble()
        var chosenAction = 0
        var runningSum = 0.0
        for (index in 0 until probabilities.shape[0]) {
            runningSum += probabilities[index]
            chosenAction = index
            if (randomValue < runningSum)
                break
        }
        return chosenAction
    }

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
