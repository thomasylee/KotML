package kotml.reinforcement.agents.tabular

import kotml.math.MutableVector
import kotml.reinforcement.RLException
import kotml.reinforcement.models.tabular.TabularModel
import kotml.reinforcement.policies.discrete.DiscreteBehaviorPolicy
import kotml.reinforcement.policies.discrete.EpsilonGreedyPolicy

class TabularQLearningAgent(
    val numStates: Int,
    val numActions: Int,
    val stepSize: Double,
    val discount: Double,
    behaviorPolicy: DiscreteBehaviorPolicy = EpsilonGreedyPolicy(),
    val model: TabularModel? = null
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

    override fun processStep(reward: Double, state: Int): Int {
        q[prevState, prevAction] += stepSize * (
            reward + discount * q(state).max()[0] - q[prevState, prevAction]
        )
        if (model != null) {
            model.observe(q, prevState, prevAction, reward, state)
            model.runIterations(q)
        }

        prevState = state
        prevAction = behaviorPolicy.chooseAction(q(state))
        return prevAction
    }

    override fun processTerminalStep(reward: Double) {
        q[prevState, prevAction] += stepSize * (
            reward - q[prevState, prevAction]
        )
        if (model != null) {
            model.observe(q, prevState, prevAction, reward, -1)
            model.runIterations(q)
        }
    }
}
