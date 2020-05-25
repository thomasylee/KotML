package kotml.reinforcement.agents

import kotlin.random.Random
import kotml.math.Vector
import kotml.regression.functions.IdentityFunction
import kotml.regression.functions.ReLU
import kotml.regression.neural.FeedforwardNeuralNetwork
import kotml.regression.neural.NeuralLayer
import kotml.regression.neural.initialization.HeInitializer
import kotml.reinforcement.functionapproximation.DeepQNetwork
import kotml.reinforcement.policies.discrete.DiscreteBehaviorPolicy
import kotml.reinforcement.policies.discrete.EpsilonGreedyPolicy
import kotml.reinforcement.replay.ExperienceReplayBuffer

class DeepQLearningAgent(
    val stateDimensions: Int,
    val dqn: DeepQNetwork,
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
        dqn = DeepQNetwork(
            network = FeedforwardNeuralNetwork(Array<NeuralLayer>(layerSizes.size) { layerIndex ->
                val numNeurons = layerSizes[layerIndex]
                val numInputs = layerSizes.getOrNull(layerIndex - 1) ?: stateDimensions
                val activationFunction =
                    if (layerIndex == layerSizes.size - 1)
                        IdentityFunction
                    else
                        ReLU
                NeuralLayer(
                    neuronCount = numNeurons,
                    activationFunction = activationFunction,
                    regressorCount = numInputs,
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
