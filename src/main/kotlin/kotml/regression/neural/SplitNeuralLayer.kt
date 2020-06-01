package kotml.regression.neural

import kotml.math.Vector
import kotml.regression.RegressionException

/**
 * `SplitNeuralLayer` contains a list of neural layer lists that track
 * parallel groupings of layers.
 *
 * Consider the following example:
 *   A -> B -----> C -> D ----> F
 *             \----> E ---/
 *
 * Layer B is a split layer with a list of (C, D) and a list of (E).
 */
class SplitNeuralLayer(
    val subLayers: List<List<NeuralLayer>>
) : NeuralLayer(
    numInputs = subLayers.firstOrNull()?.firstOrNull()?.numInputs ?: 0,
    numOutputs = subLayers.fold(0) { acc, layers ->
        acc + layers.last().numOutputs
    }
) {
    /**
     * Evaluates a vector of inputs and returns the results, with the split
     * layers flattened into a single row vector of results. The flattening
     * preserves the orders of the layers as provided in `subLayers`.
     * @param regressors input values
     * @return output values
     */
    override fun evaluate(regressors: Vector): Vector {
        val values = mutableListOf<Double>()
        (0 until subLayers.size).forEach { index ->
            var output = regressors
            subLayers[index].forEach { subLayer ->
                output = subLayer.evaluate(output)
            }
            output.forEach { values.add(it) }
        }
        return Vector(values.size) { values[it] }
    }

    /**
     * Returns a copy of the neural layer.
     * @return copy of the neural layer
     */
    override fun copy(): NeuralLayer = SplitNeuralLayer(subLayers.map { layer ->
        layer.map { subLayer -> subLayer.copy() }
    })

    /**
     * Copies the layer's weights to the weights of this layer's neurons.
     * The given layer must be the same subclass of NeuralLayer as this layer.
     * @param layer layer whose weights should be copied
     */
    override fun updateWeights(layer: NeuralLayer) {
        if (layer !is SplitNeuralLayer) {
            throw RegressionException("Cannot update weights from layer of a different type")
        }
        subLayers.forEachIndexed { layerIndex, subLayerList ->
            subLayerList.forEachIndexed { subLayerIndex, subLayer ->
                subLayer.updateWeights(layer.subLayers[layerIndex][subLayerIndex])
            }
        }
    }

    /**
     * Returns true if `other` is an equivalent `NeuralLayer`.
     * @param other nullable object to compare to this one
     * @return true if other is an equivalent NeuralLayer, false otherwise
     */
    override fun equals(other: Any?): Boolean =
        other is SplitNeuralLayer && subLayers == other.subLayers
}
