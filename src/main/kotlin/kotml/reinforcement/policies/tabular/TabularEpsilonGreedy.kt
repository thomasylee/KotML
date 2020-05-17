package kotml.reinforcement.policies.tabular

import kotlin.random.Random
import kotml.math.Vector

class TabularEpsilonGreedy(
    val epsilon: Double = 0.1,
    val random: Random = Random
) : TabularPolicy {
    override fun chooseAction(qValuesForState: Vector): Int =
        if (random.nextDouble() < epsilon)
            random.nextInt(0, qValuesForState.shape[0])
        else
            qValuesForState.argmax()

    override fun actionProbabilities(qValuesForState: Vector): Vector {
        val numActions = qValuesForState.shape[0]
        val actionsWithMaxQ = mutableListOf<Int>()
        var maxQ = -Double.MAX_VALUE
        (0 until numActions).forEach { action ->
            val qValue = qValuesForState[action]
            if (qValue > maxQ) {
                maxQ = qValue
                actionsWithMaxQ.clear()
                actionsWithMaxQ.add(action)
            } else if (qValue == maxQ) {
                actionsWithMaxQ.add(action)
            }
        }

        val nonGreedyProb = epsilon / numActions
        val greedyProb = epsilon / numActions + (1 - epsilon) / actionsWithMaxQ.size
        return Vector(numActions) { action ->
            if (actionsWithMaxQ.contains(action))
                greedyProb
            else
                nonGreedyProb
        }
    }
}
