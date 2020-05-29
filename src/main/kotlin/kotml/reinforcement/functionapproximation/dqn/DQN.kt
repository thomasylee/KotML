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
 * `DQN` stores experiences (state, action, reward, nextState) in a replay
 * buffer and samples them to update a feedforward neural network that
 * outputs action values for each action, given a particular state.
 *
 * References:
 * * Human-level control through deep reinforcement learning (2015) -
 *   Volodymyr Mnih, Koray Kavukcuoglu, David Silver, et al. -
 *   https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
 */
class DQN(
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
     * @return target Q value
     */
    override fun calculateQTarget(reward: Double, nextState: Vector): Vector {
        val evaluatedTarget = targetNetwork.evaluate(nextState)
        val nextAction = evaluatedTarget.argmax(random)
        val nextQ = Vector(numActions) { action ->
            if (action == nextAction)
                evaluatedTarget[nextAction]
            else
                0.0
        }
        return reward + discount * nextQ
    }
}
