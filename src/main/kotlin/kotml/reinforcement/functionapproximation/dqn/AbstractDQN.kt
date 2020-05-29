package kotml.reinforcement.functionapproximation.dqn

import kotlin.random.Random
import kotml.extensions.* // ktlint-disable no-wildcard-imports
import kotml.math.Vector
import kotml.regression.neural.FeedforwardNeuralNetwork
import kotml.regression.optimization.IterativeOptimizer
import kotml.reinforcement.replay.ExperienceReplayBuffer

/**
 * `AbstractDQN` is the superclass of all variants of the Deep Q-Network (DQN)
 * function approximator.
 *
 * References:
 * * Human-level control through deep reinforcement learning (2015) -
 *   Volodymyr Mnih, Koray Kavukcuoglu, David Silver, et al. -
 *   https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
 */
abstract class AbstractDQN(
    val network: FeedforwardNeuralNetwork,
    val discount: Double,
    val optimizer: IterativeOptimizer<FeedforwardNeuralNetwork, Vector>,
    val targetNetworkUpdateFrequency: Int = 10_000,
    val minibatchSize: Int = 32,
    val replayBuffer: ExperienceReplayBuffer = ExperienceReplayBuffer(),
    val random: Random = Random
) {
    val numActions = network.layers.last().neurons.size
    val targetNetwork = network.copy()
    var iterationCounter = 0

    /**
     * Returns the Q target for a non-terminal state based on its reward
     * and next state.
     * @param reward reward from taking the action
     * @param nextState state after having taken the action
     * @return target Q value
     */
    abstract fun calculateQTarget(reward: Double, nextState: Vector): Vector

    /**
     * Adds the experience to the experience replay buffer and performs
     * backpropagation on `minibatchSize` number of samples. After
     * `targetNetworkUpdateFrequency` number of weight updates, the
     * target network is updated to match the current network.
     * @param state current state
     * @param action action taken in the current state
     * @param reward reward from taking the action
     * @param nextState state after having taken the action
     * @param isTerminal true if nextState is a terminal state
     * @param random source of randomness
     */
    fun learn(
        state: Vector,
        action: Int,
        reward: Double,
        nextState: Vector?
    ) {
        replayBuffer.append(state, action, reward, nextState)

        replayBuffer.sample(minibatchSize, random).forEach { experience ->
            val target =
                if (experience.nextState == null)
                    Vector(numActions) { experience.reward }
                else
                    calculateQTarget(experience.reward, experience.nextState)

            optimizer.observe(experience.state, target)

            iterationCounter++
            if (iterationCounter >= targetNetworkUpdateFrequency) {
                iterationCounter = 0
                targetNetwork.updateWeights(network)
            }
        }
    }

    /**
     * Returns the action values for each action in a given state.
     * @param state state for which to calculate action values
     * @return action values for each action
     */
    fun evaluate(state: Vector): Vector = network.evaluate(state)
}
