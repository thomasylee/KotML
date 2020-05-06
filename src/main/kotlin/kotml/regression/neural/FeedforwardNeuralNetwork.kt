package kotml.regression.neural

import kotml.distributions.DistributionSampler
import kotml.distributions.NormalSampler
import kotml.math.Vector
import kotml.regression.optimization.WeightedOptimizer

/**
 * `FeedforwardNeuralNetwork` contains a sequence of neural layers that
 * process data in the forward direction (from input layer toward the
 * output layer).
 */
class FeedforwardNeuralNetwork(val layers: Array<NeuralLayer>) {
    /**
     * Creates a `FeedforwardNeuralNetwork` with the given number and size of
     * layers, the optimizer, and the sampler to initialize weights.
     * @param layerSizes sizes of each neural layer
     * @param optimizer optimizer copied into each neuron
     * @param sampler distribution sampler to generate initial weights
     */
    constructor(layerSizes: IntArray, optimizer: WeightedOptimizer, sampler: DistributionSampler = NormalSampler()) : this(
        Array<NeuralLayer>(layerSizes.size) { index ->
            NeuralLayer(Array<Neuron>(layerSizes[index]) {
                Neuron(optimizer.copy(sampler))
            })
        }
    )

    /**
     * Returns the results of the last neural layer in the network after
     * sending `regressors` to the first layer and processing each layer
     * with the previous layer's outputs.
     * @param regressors input values
     * @return output values from the last neural layer
     */
    fun evaluate(regressors: Vector): Vector =
        layers.fold(regressors) { input, layer ->
            layer.evaluate(input)
        }
}
