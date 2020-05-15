package kotml.reinforcement.agents.tabular

import kotlin.random.Random
import kotml.math.MutableVector
import kotml.reinforcement.RLException

class TabularExpectedSarsaAgent(
    val numStates: Int,
    val numActions: Int,
    val epsilon: Double,
    val stepSize: Double,
    val discount: Double,
    val random: Random = Random
) : TabularAgent() {
    val q = MutableVector.zeros(numStates, numActions)
    var prevState: Int = 0
    var prevAction: Int = 0

    init {
        if (numStates <= 0 || numActions <= 0)
            throw RLException("Number of states and actions must be positive")
    }

    override fun start(initialState: Int): Int {
        prevState = initialState
        prevAction = chooseAction(initialState)
        return prevAction
    }

    fun chooseAction(state: Int): Int =
        if (random.nextDouble() < epsilon)
            random.nextInt(0, numActions)
        else
            argmax(q(state))

    fun expectedQ(state: Int): Double {
        val qState = q(state)
        var actionsWithMaxQ = 0
        var maxQ = -Double.MAX_VALUE
        (0 until numActions).forEach { action ->
            val qValue = qState[action]
            if (qValue > maxQ) {
                maxQ = qValue
                actionsWithMaxQ = 1
            } else if (qValue == maxQ) {
                actionsWithMaxQ++
            }
        }
        val nonGreedyProb = epsilon / numActions
        val greedyProb = epsilon / numActions + (1 - epsilon) / actionsWithMaxQ
        return (0 until numActions).fold(0.0) { acc, action ->
            val qValue = qState[action]
            if (qValue == maxQ)
                acc + greedyProb * maxQ
            else
                acc + nonGreedyProb * qValue
        }
    }

    override fun processStep(reward: Double, state: Int): Int {
        val action = chooseAction(state)

        q[prevState, prevAction] += stepSize * (
            reward + discount * expectedQ(state) - q[prevState, prevAction]
        )

        prevState = state
        prevAction = action
        return action
    }

    override fun processTerminalStep(reward: Double) {
        q[prevState, prevAction] += stepSize * (
            reward - q[prevState, prevAction]
        )
    }
}
