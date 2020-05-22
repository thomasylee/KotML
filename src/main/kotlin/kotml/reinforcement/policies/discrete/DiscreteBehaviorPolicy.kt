package kotml.reinforcement.policies.discrete

import kotlin.random.Random
import kotml.math.Vector
import kotml.reinforcement.policies.BehaviorPolicy

abstract class DiscreteBehaviorPolicy(random: Random = Random) : BehaviorPolicy<Int>(random) {
    /**
     * Returns the chosen action based on the action values. The action
     * values should be a row vector with a length equal to the number of
     * actions.
     * @param qValuesForState action values for each action in the current state
     * @return index of the chosen action
     */
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

    /**
     * Returns a vector of probabilities that each action will be selected.
     * The action values should be a row vector with a length equal to the
     * number of actions.
     * @param qValuesForState action values for each action in the current state
     * @return index of the chosen action
     */
    abstract fun actionProbabilities(qValuesForState: Vector): Vector

    /**
     * Returns a vector of actions based on batches of action values. The
     * action values should be a vector of shape (numBatches, numActions).
     * @param qValuesForStates action values for each action in each batch
     * @return vector of chosen actions
     */
    fun batchChooseActions(qValuesForStates: Vector): Vector = Vector(qValuesForStates.shape[1]) {
        chooseAction(qValuesForStates(it)).toDouble()
    }

    /**
     * Returns a vector of probabilities that each action will be selected
     * within each batch. The action values should be a vector of shape
     * (numBatches, numActions).
     * @param qValuesForStates action values for each action in each batch
     * @return vector of action probabilities for each batch
     */
    fun batchActionProbabilities(qValuesForStates: Vector): Vector = Vector(
        *Array<Vector>(qValuesForStates.shape[0]) {
            actionProbabilities(qValuesForStates(it))
        }
    )
}
