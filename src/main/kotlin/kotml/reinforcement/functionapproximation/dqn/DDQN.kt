package kotml.reinforcement.functionapproximation.dqn

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
 * `DDQN` implements the Double Deep Q-Network (DDQN) function approximator
 * that addresses DQN's tendency to sometimes overestimate action values,
 * particularly as the number of actions increases.
 *
 * References:
 * * Deep reinforcement learning with double Q-learning (2015) -
 *   Hado van Hasselt, Arthur Guez, David Silver
 *   https://arxiv.org/pdf/1509.06461.pdf
 */
class DDQN(
    network: FeedforwardNeuralNetwork,
    discount: Double,
    optimizer: IterativeOptimizer<FeedforwardNeuralNetwork, Vector>,
    targetNetworkUpdateFrequency: Int = 10_000,
    minibatchSize: Int = 32,
    replayBuffer: ExperienceReplayBuffer = ExperienceReplayBuffer(),
    random: Random = Random
) : AbstractDQN(
    network = network,
    discount = discount,
    optimizer = optimizer,
    targetNetworkUpdateFrequency = targetNetworkUpdateFrequency,
    minibatchSize = minibatchSize,
    replayBuffer = replayBuffer,
    random = random
) {
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
     * Returns the Q target for a non-terminal state based on its reward
     * and next state.
     * @param reward reward from taking the action
     * @param nextState state after having taken the action
     * @return target Q value at the next action's index, zeros at other indices
     */
    override fun calculateQTarget(reward: Double, nextState: Vector): Vector {
        // Y_t = R_t+1 + discnt * Q(S_t+1, argmax Q(S_t+1, a, theta), theta-)
        // where theta is online network and theta- is target network.
        val evaluatedTarget = network.evaluate(nextState)
        val nextAction = evaluatedTarget.argmax(random)
        val nextQ = targetNetwork.evaluate(nextState).mapIndexed { action, q ->
            if (action == nextAction)
                q
            else
                0.0
        }
        return reward + discount * nextQ
    }
}
