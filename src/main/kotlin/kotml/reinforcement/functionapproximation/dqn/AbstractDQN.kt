package kotml.reinforcement.functionapproximation.dqn

import kotlin.random.Random
import kotml.distributions.UniformSampler
import kotml.math.Vector
import kotml.regression.functions.IdentityFunction
import kotml.regression.functions.ReLU
import kotml.regression.neural.DenseNeuralLayer
import kotml.regression.neural.FeedforwardNeuralNetwork
import kotml.regression.neural.Neuron
import kotml.regression.neural.SplitNeuralLayer
import kotml.regression.neural.initialization.HeInitializer
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
    val numActions = network.layers.last().numOutputs
    val targetNetwork = network.copy()
    var iterationCounter = 0

    /**
     * Returns the Q target for a non-terminal state based on its reward
     * and next state.
     * @param reward reward from taking the action
     * @param nextState state after having taken the action
     * @return target Q value at the next action's index, zeros at other indices
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

    companion object {
        /**
         * Returns a `FeedforwardNeuralNetwork` matching the usual dense
         * neural network that DQN uses, with two layers: ReLU activation
         * functions for the first and identity activation functions for the
         * second.
         * @param stateDimensions number of state dimensions (neural network inputs)
         * @param numActions number of actions (neural network outputs)
         * @return neural network with identity and ReLU activation functions
         */
        @JvmStatic
        fun createNeuralNetwork(
            stateDimensions: Int,
            numActions: Int,
            random: Random = Random
        ): FeedforwardNeuralNetwork =
            createNeuralNetwork(stateDimensions, intArrayOf(stateDimensions, numActions), random)

        /**
         * Returns a `FeedforwardNeuralNetwork` matching the usual dense neural
         * network that DQN uses, with ReLU activation functions for all layers
         * except the last, which uses the identity activation function.
         * @param stateDimensions number of state dimensions (neural network inputs)
         * @param layerSizes sizes of the layers
         * @return neural network with identity and ReLU activation functions
         */
        @JvmStatic
        fun createNeuralNetwork(
            stateDimensions: Int,
            layerSizes: IntArray,
            random: Random = Random
        ): FeedforwardNeuralNetwork =
            FeedforwardNeuralNetwork(layerSizes.mapIndexed { layerIndex, numNeurons ->
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
            })

        /**
         * Returns a `FeedforwardNeuralNetwork` structured as a dueling neural
         * network, as described in "Dueling Network Architectures for Deep
         * Reinforcement Learning" (2015) by Ziyu Wang, et al. The common layers
         * use ReLU, and the state value and advantage value layers use ReLU
         * for all except the last layers in each, which use identity functions.
         * The single `SplitNeuralLayer` has the state value layers as the first
         * sublayer list and the advantage value layers as the second sublayer
         * list. The final aggregating layer adds the state and advantage outputs
         * while subtracting the average advantage, as used in equation (9) in
         * the aforementioned research paper.
         * @param stateDimensions number of state dimensions (neural network inputs)
         * @param numActions number of actions (neural network outputs)
         * @param commonLayerSizes sizes of the common layers (before splitting
         *   into separate state and advantage layers)
         * @param stateValueLayerSizes sizes of the state value layers
         * @param advantageValueLayerSizes sizes of the advantage value layers
         * @return dueling neural network
         */
        @JvmStatic
        fun createDuelingNeuralNetwork(
            stateDimensions: Int,
            numActions: Int,
            commonLayerSizes: IntArray = intArrayOf(stateDimensions),
            stateValueLayerSizes: IntArray = intArrayOf(1),
            advantageValueLayerSizes: IntArray = intArrayOf(numActions),
            random: Random = Random
        ): FeedforwardNeuralNetwork = FeedforwardNeuralNetwork(
            commonLayerSizes.mapIndexed { layerIndex, numNeurons ->
                val numInputs = commonLayerSizes.getOrNull(layerIndex - 1) ?: stateDimensions
                DenseNeuralLayer(
                    numInputs = numInputs,
                    neuronCount = numNeurons,
                    activationFunction = ReLU,
                    sampler = HeInitializer.sampler(numInputs, numNeurons, random)
                )
            } + listOf(SplitNeuralLayer(listOf(
                stateValueLayerSizes.mapIndexed { layerIndex, numNeurons ->
                    val numInputs = stateValueLayerSizes.getOrNull(layerIndex - 1) ?: commonLayerSizes.last()
                    val activationFunction =
                        if (layerIndex == stateValueLayerSizes.size - 1)
                            IdentityFunction
                        else
                            ReLU
                    DenseNeuralLayer(
                        numInputs = numInputs,
                        neuronCount = numNeurons,
                        activationFunction = activationFunction,
                        sampler = HeInitializer.sampler(numInputs, numNeurons, random)
                    )
                },
                advantageValueLayerSizes.mapIndexed { layerIndex, numNeurons ->
                    val numInputs = advantageValueLayerSizes.getOrNull(layerIndex - 1) ?: commonLayerSizes.last()
                    val activationFunction =
                        if (layerIndex == advantageValueLayerSizes.size - 1)
                            IdentityFunction
                        else
                            ReLU
                    DenseNeuralLayer(
                        numInputs = numInputs,
                        neuronCount = numNeurons,
                        activationFunction = activationFunction,
                        sampler = HeInitializer.sampler(numInputs, numNeurons, random)
                    )
                }
            ))) + listOf(DenseNeuralLayer((0 until numActions).map { action ->
                Neuron(
                    activationFunction = IdentityFunction,
                    regressorCount = numActions + 1,
                    includeConstant = false,
                    sampler = UniformSampler(0.0),
                    aggregationFunction = DuelingAggregationFunction(action)
                )
            }))
        )
    }
}
