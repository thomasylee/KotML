package kotml.regression.neural

import kotml.extensions.* // ktlint-disable no-wildcard-imports
import kotml.math.Vector
import kotml.regression.RegressionException
import kotml.regression.functions.FunctionModel
import kotml.regression.functions.PReLU
import kotml.regression.functions.ReLU
import kotml.regression.functions.aggregation.AggregationFunction
import kotml.regression.functions.aggregation.DotProduct
import kotml.regression.neural.initialization.HeInitializer
import kotml.regression.neural.initialization.NeuralNetworkInitializer
import kotml.regression.neural.initialization.XavierInitializer

/**
 * `FeedforwardNeuralNetwork` contains a sequence of neural layers that
 * process data in the forward direction (from input layer toward the
 * output layer).
 */
class FeedforwardNeuralNetwork(
    val stepSize: Double,
    val layers: Array<NeuralLayer>
) {
    init {
        if (layers.isEmpty())
            throw RegressionException("Neural networks must have at least one neural layer")
    }

    /**
     * Creates a `FeedforwardNeuralNetwork` with the given number and size of
     * layers, the optimizer, and the sampler to initialize weights.
     * @param layerSizes sizes of each neural layer
     * @param optimizer optimizer copied into each neuron
     * @param sampler distribution sampler to generate initial weights
     */
    constructor(
        stepSize: Double,
        inputCount: Int,
        layerSizes: IntArray,
        activationFunction: FunctionModel,
        aggregationFunction: AggregationFunction = DotProduct,
        includeConstant: Boolean = true,
        initializer: NeuralNetworkInitializer =
            if (activationFunction is ReLU || activationFunction is PReLU)
                HeInitializer
            else
                XavierInitializer
    ) : this(
        stepSize,
        Array<NeuralLayer>(layerSizes.size) { index ->
            val regressorCount = layerSizes.getOrElse(index - 1) { inputCount }
            NeuralLayer(Array<Neuron>(layerSizes[index]) {
                Neuron(
                    activationFunction = activationFunction,
                    regressorCount = regressorCount,
                    includeConstant = includeConstant,
                    sampler = initializer.sampler(regressorCount, layerSizes[index]),
                    aggregationFunction = aggregationFunction
                )
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

    /**
     * Returns a copy of the neural network.
     * @return copy of the neural network
     */
    fun copy(): FeedforwardNeuralNetwork = FeedforwardNeuralNetwork(
        stepSize = stepSize,
        layers = Array<NeuralLayer>(layers.size) { layers[it].copy() }
    )

    /**
     * Copies the weights of `network` into this network's neurons.
     * @param network network whose weights should be copied
     */
    fun updateWeights(network: FeedforwardNeuralNetwork) {
        layers.forEachIndexed { index, layer ->
            layer.updateWeights(network.layers[index])
        }
    }

    /**
     * Returns true if `other` is an equivalent `FeedforwardNeuralNetwork`.
     * @param other nullable object to compare to this one
     * @return true if other is an equivalent FeedforwardNeuralNetwork, or
     *   false otherwise
     */
    override fun equals(other: Any?): Boolean =
        other is FeedforwardNeuralNetwork && layers.contentEquals(other.layers)
}
