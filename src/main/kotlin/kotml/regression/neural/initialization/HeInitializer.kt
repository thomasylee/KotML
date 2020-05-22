package kotml.regression.neural.initialization

import kotlin.math.sqrt
import kotlin.random.Random
import kotml.distributions.DistributionSampler
import kotml.distributions.NormalSampler

/**
 * He initialization sets initial weights for a neural network by
 * sampling from a normal distribution with mean 0 and variance equal to
 * 2 divided by the number of inputs.
 *
 * He initialization tends to work well with `ReLU` activation functions.
 */
object HeInitializer : NeuralNetworkInitializer {
    /**
     * Returns the `DistributionSampler` that should be used to initialize
     * weights with the given number of inputs and outputs.
     * @param numInputs number of inputs
     * @param numOutputs number of outputs
     * @param random source of randomness
     * @return sampler for initializing weights in a neural network layer
     */
    override fun sampler(numInputs: Int, numOutputs: Int, random: Random): DistributionSampler =
        NormalSampler(
            mean = 0.0,
            stdev = sqrt(2.0 / numInputs),
            random = random
        )
}
