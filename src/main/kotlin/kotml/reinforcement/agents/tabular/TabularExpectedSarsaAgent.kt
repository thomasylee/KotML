package kotml.reinforcement.agents.tabular

import kotml.math.MutableVector
import kotml.reinforcement.RLException
import kotml.reinforcement.policies.tabular.TabularEpsilonGreedy
import kotml.reinforcement.policies.tabular.TabularPolicy

class TabularExpectedSarsaAgent(
    val numStates: Int,
    val numActions: Int,
    val stepSize: Double,
    val discount: Double,
    behaviorPolicy: TabularPolicy = TabularEpsilonGreedy()
) : TabularAgent(behaviorPolicy) {
    val q = MutableVector.zeros(numStates, numActions)
    var prevState: Int = 0
    var prevAction: Int = 0

    init {
        if (numStates <= 0 || numActions <= 0)
            throw RLException("Number of states and actions must be positive")
    }

    override fun start(initialState: Int): Int {
        prevState = initialState
        prevAction = behaviorPolicy.chooseAction(q(initialState))
        return prevAction
    }

    fun expectedQ(state: Int): Double {
        val actionProb = behaviorPolicy.actionProbabilities(q(state))
        return (actionProb * q(state)).sum()[0]
    }

    override fun processStep(reward: Double, state: Int): Int {
        val action = behaviorPolicy.chooseAction(q(state))

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
