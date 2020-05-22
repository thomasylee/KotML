package kotml.reinforcement.agents.tabular

import kotml.math.MutableVector
import kotml.reinforcement.RLException
import kotml.reinforcement.models.tabular.TabularModel
import kotml.reinforcement.policies.discrete.DiscreteBehaviorPolicy
import kotml.reinforcement.policies.discrete.EpsilonGreedyPolicy

class TabularExpectedSarsaAgent(
    val numStates: Int,
    val numActions: Int,
    val stepSize: Double,
    val discount: Double,
    behaviorPolicy: DiscreteBehaviorPolicy = EpsilonGreedyPolicy(),
    // TODO: Mutating this after initialization is pretty wonky.
    // Creating and assigning the model only once while still allowing it
    // to calculated expected Q values would be preferable.
    var model: TabularModel? = null
) : TabularAgent(behaviorPolicy) {
    val q = MutableVector.zeros(numStates, numActions)
    val discreteBehaviorPolicy: DiscreteBehaviorPolicy
    var prevState: Int = 0
    var prevAction: Int = 0

    init {
        if (numStates <= 0 || numActions <= 0)
            throw RLException("Number of states and actions must be positive")

        discreteBehaviorPolicy = behaviorPolicy
    }

    override fun start(initialState: Int): Int {
        prevState = initialState
        prevAction = discreteBehaviorPolicy.chooseAction(q(initialState))
        return prevAction
    }

    fun expectedQ(state: Int): Double {
        val actionProb = discreteBehaviorPolicy.actionProbabilities(q(state))
        return (actionProb * q(state)).sum()[0]
    }

    override fun processStep(reward: Double, state: Int): Int {
        q[prevState, prevAction] += stepSize * (
            reward + discount * expectedQ(state) - q[prevState, prevAction]
        )

        val constModel = model
        if (constModel != null) {
            constModel.observe(q, prevState, prevAction, reward, state)
            constModel.runIterations(q)
        }

        prevState = state
        prevAction = discreteBehaviorPolicy.chooseAction(q(state))
        return prevAction
    }

    override fun processTerminalStep(reward: Double) {
        q[prevState, prevAction] += stepSize * (
            reward - q[prevState, prevAction]
        )

        val constModel = model
        if (constModel != null) {
            constModel.observe(q, prevState, prevAction, reward, -1)
            constModel.runIterations(q)
        }
    }
}
