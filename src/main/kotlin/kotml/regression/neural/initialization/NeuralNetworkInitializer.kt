package kotml.regression.neural.initialization

import kotlin.random.Random
import kotml.distributions.DistributionSampler

/**
 * `NeuralNetworkInitializer` initializes the weights of a neural network
 * according to a policy that takes into account the number of inputs and
 * outputs for each layer.
 */
interface NeuralNetworkInitializer {
    /**
     * Returns the `DistributionSampler` that should be used to initialize
     * weights with the given number of inputs and outputs.
     * @param numInputs number of inputs
     * @param numOutputs number of outputs
     * @return sampler for initializing weights in a neural network layer
     */
    fun sampler(numInputs: Int, numOutputs: Int): DistributionSampler =
        sampler(numInputs, numOutputs, Random)

    /**
     * Returns the `DistributionSampler` that should be used to initialize
     * weights with the given number of inputs and outputs.
     * @param numInputs number of inputs
     * @param numOutputs number of outputs
     * @param random source of randomness
     * @return sampler for initializing weights in a neural network layer
     */
    abstract fun sampler(numInputs: Int, numOutputs: Int, random: Random): DistributionSampler
}
