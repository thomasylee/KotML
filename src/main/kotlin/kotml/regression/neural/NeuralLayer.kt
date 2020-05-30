package kotml.regression.neural

import kotml.math.Vector

/**
 * `NeuralLayer` contains a collection of neurons to be used in a neural
 * network.
 */
abstract class NeuralLayer(val numInputs: Int, val numOutputs: Int) {
    /**
     * Evaluates a vector of inputs and returns the results.
     * @param regressors input values
     * @return output values
     */
    abstract fun evaluate(regressors: Vector): Vector

    /**
     * Returns a copy of the neural layer.
     * @return copy of the neural layer
     */
    abstract fun copy(): NeuralLayer

    /**
     * Copies the layer's weights to the weights of this layer's neurons.
     * The given layer must be the same subclass of NeuralLayer as this layer.
     * @param layer layer whose weights should be copied
     */
    abstract fun updateWeights(layer: NeuralLayer)

    /**
     * Returns true if `other` is an equivalent `NeuralLayer`.
     * @param other nullable object to compare to this one
     * @return true if other is an equivalent NeuralLayer, false otherwise
     */
    abstract override fun equals(other: Any?): Boolean
}
