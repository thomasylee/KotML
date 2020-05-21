package kotml.reinforcement.models.tabular

import java.util.PriorityQueue
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random
import kotml.math.MutableVector

/**
 * `PrioritizedDynaQPlus` builds a model that prioritizes state updates with
 * the highest expected TD error to facilitate quicker convergence to the
 * true action values.
 */
class PrioritizedDynaQPlus(
    numIterations: Int,
    val numStates: Int,
    val numActions: Int,
    val qUpdate: (prevState: Int, prevAction: Int, reward: Double, state: Int, qValues: MutableVector) -> Double,
    val kappa: Double = 0.001,
    val priorityThreshold: Double = 0.0,
    val maxQueueSize: Int = 10_000,
    val random: Random = Random
) : TabularModel(numIterations) {
    val model = mutableMapOf<StateAndAction, RewardAndState>()
    val timeSinceLastVisit = MutableVector.zeros(numStates, numActions)
    val statesLeadingToState = mutableMapOf<Int, MutableList<Int>>()
    val actionsByState = mutableMapOf<Int, MutableList<Int>>()
    val priorityQueue = PriorityQueue<PrioritizedStateAndAction>()
    val reversePriorityQueue = PriorityQueue<PrioritizedStateAndAction>(object : Comparator<PrioritizedStateAndAction> {
        override fun compare(first: PrioritizedStateAndAction, second: PrioritizedStateAndAction): Int =
            second.priority.compareTo(first.priority)
    })
    val inQueue = mutableMapOf<StateAndAction, Double>()

    data class StateAndAction(val state: Int, val action: Int)
    data class PrioritizedStateAndAction(
        val state: Int,
        val action: Int,
        val priority: Double
    ) : Comparable<PrioritizedStateAndAction> {
        override operator fun compareTo(other: PrioritizedStateAndAction): Int =
            priority.compareTo(other.priority)
    }
    data class RewardAndState(val reward: Double, val state: Int)

    override fun observe(qValues: MutableVector, prevState: Int, prevAction: Int, reward: Double, state: Int) {
        model.put(StateAndAction(prevState, prevAction), RewardAndState(reward, state))
        timeSinceLastVisit += 1
        timeSinceLastVisit[prevState, prevAction] = 0

        val leadingStates = statesLeadingToState.getOrPut(state) {
            mutableListOf<Int>()
        }
        if (!leadingStates.contains(prevState))
            leadingStates.add(prevState)

        val actions = actionsByState.getOrPut(prevState) { mutableListOf<Int>() }
        if (!actions.contains(prevAction))
            actions.add(prevAction)

        enqueueIfHighPriority(prevState, prevAction, reward, state, qValues)
    }

    override fun iterate(qValues: MutableVector): Boolean {
        if (priorityQueue.isEmpty())
            return false

        val prioritizedStateAndAction = priorityQueue.poll()
        reversePriorityQueue.remove(prioritizedStateAndAction)
        val stateAndAction = StateAndAction(
            prioritizedStateAndAction.state, prioritizedStateAndAction.action)
        inQueue.remove(stateAndAction)

        val rewardAndState = model.get(stateAndAction)
        if (rewardAndState == null)
            return true

        val reward = rewardAndState.reward + kappa * sqrt(timeSinceLastVisit[
            stateAndAction.state, stateAndAction.action])

        qValues[stateAndAction.state, stateAndAction.action] +=
            qUpdate(stateAndAction.state, stateAndAction.action,
                reward, rewardAndState.state, qValues)

        statesLeadingToState[stateAndAction.state]?.forEach { formerState ->
            actionsByState[formerState]?.forEach { formerAction ->
                val formerRewardAndState = model[StateAndAction(formerState, formerAction)]
                val formerReward = formerRewardAndState?.reward
                val formerNextState = formerRewardAndState?.state
                if (formerReward != null && formerNextState == stateAndAction.state && formerState != stateAndAction.state) {
                    enqueueIfHighPriority(
                        formerState,
                        formerAction,
                        formerReward + kappa * sqrt(timeSinceLastVisit[formerState, formerAction]),
                        formerNextState,
                        qValues
                    )
                }
            }
        }

        return !priorityQueue.isEmpty()
    }

    private fun enqueueIfHighPriority(prevState: Int, prevAction: Int, reward: Double, state: Int, qValues: MutableVector) {
        val updatedQ = qUpdate(prevState, prevAction, reward, state, qValues)
        val priority = abs(updatedQ)
        if (priority > priorityThreshold) {
            val stateAndAction = StateAndAction(prevState, prevAction)
            val inQueuePriority = inQueue[stateAndAction]
            if (inQueuePriority != null && inQueuePriority > priority)
                return

            if (inQueuePriority != null) {
                val toRemove = PrioritizedStateAndAction(prevState, prevAction, inQueuePriority)
                priorityQueue.remove(toRemove)
                reversePriorityQueue.remove(toRemove)
                inQueue.remove(stateAndAction)
            }

            val toAdd = PrioritizedStateAndAction(prevState, prevAction, priority)
            priorityQueue.add(toAdd)
            reversePriorityQueue.add(toAdd)
            inQueue.put(stateAndAction, priority)

            if (priorityQueue.size > maxQueueSize) {
                val toRemove = reversePriorityQueue.poll()
                priorityQueue.remove(toRemove)
                inQueue.remove(StateAndAction(toRemove.state, toRemove.action))
            }
        }
    }
}
