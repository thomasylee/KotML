package kotml.reinforcement.functionapproximation

import kotlin.random.Random
import kotml.extensions.* // ktlint-disable no-wildcard-imports
import kotml.math.Vector
import kotml.regression.cost.SumCost
import kotml.regression.cost.loss.HalfSquaredError
import kotml.regression.neural.FeedforwardNeuralNetwork
import kotml.regression.optimization.IterativeOptimizer
import kotml.regression.optimization.backpropagation.AdamBackpropagation
import kotml.reinforcement.replay.ExperienceReplayBuffer

/**
 * `DeepQNetwork` stores experiences (state, action, reward, nextState,
 * isTerminal) in a replay buffer and samples them to update a feedforward
 * neural network that outputs action values for each action, given a
 * particular state.
 *
 * References:
 * * Human-level control through deep reinforcement learning (2015) -
 *   Volodymyr Mnih, Koray Kavukcuoglu, David Silver, et al. -
 *   https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
 */
class DeepQNetwork(
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

    constructor(
        network: FeedforwardNeuralNetwork,
        discount: Double,
        stepSize: Double,
        targetNetworkUpdateFrequency: Int = 10_000,
        minibatchSize: Int = 32,
        replayBuffer: ExperienceReplayBuffer = ExperienceReplayBuffer(),
        random: Random = Random
    ) : this(
        network = network,
        discount = discount,
        optimizer = AdamBackpropagation(
            network = network,
            costFunction = SumCost(HalfSquaredError),
            stepSize = stepSize
        ),
        targetNetworkUpdateFrequency = targetNetworkUpdateFrequency,
        minibatchSize = minibatchSize,
        replayBuffer = replayBuffer,
        random = random
    )

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
                if (experience.nextState == null) {
                    Vector(numActions) { experience.reward }
                } else {
                    val evaluatedTarget = targetNetwork.evaluate(experience.nextState)
                    val nextAction = evaluatedTarget.argmax(random)
                    val nextQ = Vector(numActions) { action ->
                        if (action == nextAction)
                            evaluatedTarget[nextAction]
                        else
                            0.0
                    }
                    experience.reward + discount * nextQ
                }

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
