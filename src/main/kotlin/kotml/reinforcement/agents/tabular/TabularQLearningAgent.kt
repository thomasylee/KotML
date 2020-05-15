package kotml.reinforcement.agents.tabular

import kotlin.random.Random
import kotml.math.MutableVector
import kotml.reinforcement.RLException

class TabularQLearningAgent(
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

    override fun processStep(reward: Double, state: Int): Int {
        val action = chooseAction(state)

        q[prevState, prevAction] += stepSize * (
            reward + discount * q(state).max()[0] - q[prevState, prevAction]
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
