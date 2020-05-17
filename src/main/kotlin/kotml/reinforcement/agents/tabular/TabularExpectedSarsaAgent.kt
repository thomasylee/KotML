package kotml.reinforcement.agents.tabular

import kotml.math.MutableVector
import kotml.reinforcement.RLException
import kotml.reinforcement.models.tabular.TabularModel
import kotml.reinforcement.policies.tabular.TabularEpsilonGreedy
import kotml.reinforcement.policies.tabular.TabularPolicy

class TabularExpectedSarsaAgent(
    val numStates: Int,
    val numActions: Int,
    val stepSize: Double,
    val discount: Double,
    behaviorPolicy: TabularPolicy = TabularEpsilonGreedy(),
    // TODO: Mutating this after initialization is pretty wonky.
    // Creating and assigning the model only once while still allowing it
    // to calculated expected Q values would be preferable.
    var model: TabularModel? = null
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
        q[prevState, prevAction] += stepSize * (
            reward + discount * expectedQ(state) - q[prevState, prevAction]
        )

        val constModel = model
        if (constModel != null) {
            constModel.observe(q, prevState, prevAction, reward, state)
            constModel.runIterations(q)
        }

        prevState = state
        prevAction = behaviorPolicy.chooseAction(q(state))
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
