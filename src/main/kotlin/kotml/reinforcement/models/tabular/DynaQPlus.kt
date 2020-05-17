package kotml.reinforcement.models.tabular

import kotlin.math.sqrt
import kotlin.random.Random
import kotml.math.MutableVector

/**
 * `DynaQPlus` updates Q values according to the DynaQ+ planning algorithm.
 */
class DynaQPlus(
    numIterations: Int,
    val numStates: Int,
    val numActions: Int,
    val qUpdate: (prevState: Int, prevAction: Int, reward: Double, state: Int, qValues: MutableVector) -> Double,
    val kappa: Double = 0.001,
    val random: Random = Random
) : TabularModel(numIterations) {
    val model = mutableMapOf<StateAndAction, RewardAndState>()
    val timeSinceLastVisit = MutableVector.zeros(numStates, numActions)

    data class StateAndAction(val state: Int, val action: Int)
    data class RewardAndState(val reward: Double, val state: Int)

    /**
     * Performs model learning using the provided qValues and state
     * transition.
     * @param qValues mutable vector of Q values
     * @param prevState previous state
     * @param prevAction previous action
     * @param reward reward of the state transition
     * @param state new state
     */
    override fun observe(qValues: MutableVector, prevState: Int, prevAction: Int, reward: Double, state: Int) {
        model.put(StateAndAction(prevState, prevAction), RewardAndState(reward, state))
        timeSinceLastVisit += 1
        timeSinceLastVisit[prevState, prevAction] = 0
    }

    /**
     * Performs one planning step by mutating values in qValues.
     * @param qValues mutable vector of Q values
     * @return true if more iterations should run, false otherwise
     */
    override fun iterate(qValues: MutableVector): Boolean {
        val stateAndAction = model.keys.random(random)
        val rewardAndState = model.get(stateAndAction)
        if (rewardAndState == null)
            return true

        // Apply bonus reward based on how long ago the stage was visited.
        val reward = rewardAndState.reward + kappa * sqrt(timeSinceLastVisit[
            stateAndAction.state, stateAndAction.action])

        qValues[stateAndAction.state, stateAndAction.action] +=
            qUpdate(stateAndAction.state, stateAndAction.action,
                reward, rewardAndState.state, qValues)

        return true
    }
}
