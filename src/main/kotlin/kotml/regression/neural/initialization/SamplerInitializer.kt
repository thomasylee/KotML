package kotml.regression.neural.initialization

import kotlin.random.Random
import kotml.distributions.DistributionSampler

/**
 * `SamplerInitializer` uses provided sampler to initialize neural network
 * weights.
 */
class SamplerInitializer(val sampler: DistributionSampler) : NeuralNetworkInitializer {
    /**
     * Returns this SamplerInitializer's sampler.
     * @param numInputs (ignored) number of inputs
     * @param numOutputs (ignored) number of outputs
     * @param random (ignored) source of randomness
     * @return this SamplerInitializer's sampler
     */
    override fun sampler(numInputs: Int, numOutputs: Int, random: Random): DistributionSampler = sampler
}
