package kotml.reinforcement.agents

import kotlin.random.Random
import kotml.math.Vector
import kotml.regression.functions.IdentityFunction
import kotml.regression.functions.ReLU
import kotml.regression.neural.DenseNeuralLayer
import kotml.regression.neural.FeedforwardNeuralNetwork
import kotml.regression.neural.initialization.HeInitializer
import kotml.reinforcement.functionapproximation.dqn.AbstractDQN
import kotml.reinforcement.functionapproximation.dqn.DQN
import kotml.reinforcement.policies.discrete.DiscreteBehaviorPolicy
import kotml.reinforcement.policies.discrete.EpsilonGreedyPolicy
import kotml.reinforcement.replay.ExperienceReplayBuffer

/**
 * `DeepQLearningAgent` uses a deep Q-network as an action value function
 * approximator, following traditional Q learning action selection using
 * the Q-network's estimated action values.
 *
 * References:
 * * Human-level control through deep reinforcement learning (2015) -
 *   Volodymyr Mnih, Koray Kavukcuoglu, David Silver, et al. -
 *   https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
 */
class DeepQLearningAgent(
    val stateDimensions: Int,
    val dqn: AbstractDQN,
    behaviorPolicy: DiscreteBehaviorPolicy = EpsilonGreedyPolicy(),
    val random: Random = Random
) : RLAgent<Vector, Int>(behaviorPolicy) {
    var prevState: Vector = Vector.zeros(1)
    var prevAction: Int = 0

    constructor(
        stateDimensions: Int,
        discount: Double,
        stepSize: Double,
        layerSizes: IntArray,
        behaviorPolicy: DiscreteBehaviorPolicy = EpsilonGreedyPolicy(),
        targetNetworkUpdateFrequency: Int = 10_000,
        minibatchSize: Int = 32,
        replayBuffer: ExperienceReplayBuffer = ExperienceReplayBuffer(),
        random: Random = Random
    ) : this(
        behaviorPolicy = behaviorPolicy,
        stateDimensions = stateDimensions,
        dqn = DQN(
            network = FeedforwardNeuralNetwork(layerSizes.mapIndexed { layerIndex, numNeurons ->
                val numInputs = layerSizes.getOrNull(layerIndex - 1) ?: stateDimensions
                val activationFunction =
                    if (layerIndex == layerSizes.size - 1)
                        IdentityFunction
                    else
                        ReLU
                DenseNeuralLayer(
                    numInputs = numInputs,
                    neuronCount = numNeurons,
                    activationFunction = activationFunction,
                    sampler = HeInitializer.sampler(numInputs, numNeurons, random)
                )
            }),
            discount = discount,
            stepSize = stepSize,
            targetNetworkUpdateFrequency = targetNetworkUpdateFrequency,
            minibatchSize = minibatchSize,
            replayBuffer = replayBuffer,
            random = random
        ),
        random = random
    )

    override fun start(initialState: Vector): Int {
        prevState = initialState
        prevAction = behaviorPolicy.chooseAction(dqn.evaluate(initialState))
        return prevAction
    }

    override fun processStep(reward: Double, state: Vector): Int {
        dqn.learn(prevState, prevAction, reward, state)
        prevState = state
        prevAction = behaviorPolicy.chooseAction(dqn.evaluate(state))
        return prevAction
    }

    override fun processTerminalStep(reward: Double) {
        dqn.learn(prevState, prevAction, reward, null)
    }
}
