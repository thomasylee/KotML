package kotml.regression.neural

import kotlin.random.Random
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
    val layers: List<NeuralLayer>
) {
    val prevDenseLayers: Map<DenseNeuralLayer, List<DenseNeuralLayer>>
    val nextDenseLayers: Map<DenseNeuralLayer, List<DenseNeuralLayer>>
    val denseLayers: List<DenseNeuralLayer>
    val lastDenseLayers: List<DenseNeuralLayer>

    init {
        if (layers.isEmpty())
            throw RegressionException("Neural networks must have at least one neural layer")

        val prevDenseLayers = mutableMapOf<DenseNeuralLayer, MutableList<DenseNeuralLayer>>()
        val nextDenseLayers = mutableMapOf<DenseNeuralLayer, MutableList<DenseNeuralLayer>>()
        val denseLayers = mutableListOf<DenseNeuralLayer>()

        var prevLayers = listOf<DenseNeuralLayer>()
        layers.forEach { layer ->
            prevLayers = extractDenseLayers(
                layer, prevLayers, denseLayers, prevDenseLayers, nextDenseLayers
            )
        }

        this.lastDenseLayers = prevLayers
        this.denseLayers = denseLayers
        this.prevDenseLayers = prevDenseLayers
        this.nextDenseLayers = nextDenseLayers
    }

    private fun extractDenseLayers(
        layer: NeuralLayer,
        prevLayers: List<DenseNeuralLayer>,
        denseLayers: MutableList<DenseNeuralLayer>,
        prevDenseLayers: MutableMap<DenseNeuralLayer, MutableList<DenseNeuralLayer>>,
        nextDenseLayers: MutableMap<DenseNeuralLayer, MutableList<DenseNeuralLayer>>
    ): List<DenseNeuralLayer> {
        if (layer is SplitNeuralLayer) {
            val lastSubLayers = mutableListOf<DenseNeuralLayer>()
            layer.subLayers.forEach { subLayerList ->
                var prevSubLayers = prevLayers
                subLayerList.forEach { subLayer ->
                    prevSubLayers = extractDenseLayers(
                        subLayer, prevSubLayers, denseLayers,
                        prevDenseLayers, nextDenseLayers
                    )
                }
                lastSubLayers.addAll(prevSubLayers)
            }
            return lastSubLayers
        } else if (layer !is DenseNeuralLayer) {
            throw RegressionException("Cannot backpropagate over unknown layer type: ${layer::class}")
        }

        denseLayers.add(layer)

        prevDenseLayers.put(layer, prevLayers.toMutableList())
        prevLayers.forEach { prevLayer ->
            nextDenseLayers.getOrPut(prevLayer) {
                mutableListOf()
            }.add(layer)
        }

        return listOf(layer)
    }

    /**
     * Creates a `FeedforwardNeuralNetwork` with the given number and size of
     * layers, the optimizer, and the sampler to initialize weights.
     * @param layerSizes sizes of each neural layer
     * @param optimizer optimizer copied into each neuron
     * @param sampler distribution sampler to generate initial weights
     */
    constructor(
        inputCount: Int,
        layerSizes: IntArray,
        activationFunction: FunctionModel,
        aggregationFunction: AggregationFunction = DotProduct,
        includeConstant: Boolean = true,
        initializer: NeuralNetworkInitializer =
            if (activationFunction is ReLU || activationFunction is PReLU)
                HeInitializer
            else
                XavierInitializer,
        random: Random = Random
    ) : this(
        layerSizes.mapIndexed { index, layerSize ->
            val regressorCount = layerSizes.getOrElse(index - 1) { inputCount }
            DenseNeuralLayer((0 until layerSize).map {
                Neuron(
                    activationFunction = activationFunction,
                    regressorCount = regressorCount,
                    includeConstant = includeConstant,
                    sampler = initializer.sampler(regressorCount, layerSizes[index], random),
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
        layers.map { it.copy() }
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
        other is FeedforwardNeuralNetwork && layers == other.layers
}
