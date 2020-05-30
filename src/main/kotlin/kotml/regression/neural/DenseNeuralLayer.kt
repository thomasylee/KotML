package kotml.regression.neural

import kotml.distributions.DistributionSampler
import kotml.distributions.NormalSampler
import kotml.distributions.UniformSampler
import kotml.math.Vector
import kotml.regression.RegressionException
import kotml.regression.functions.FunctionModel
import kotml.regression.functions.IdentityFunction
import kotml.regression.functions.aggregation.AggregationFunction
import kotml.regression.functions.aggregation.DotProduct
import kotml.regression.functions.aggregation.Softmax

/**
 * `DenseNeuralLayer` contains a collection of densely packed neurons to be
 * used in a neural network.
 */
class DenseNeuralLayer(val neurons: List<Neuron>) : NeuralLayer(
    numInputs = neurons.firstOrNull()?.regressorCount ?: 0,
    numOutputs = neurons.size
) {
    companion object {
        /**
         * Returns a `DenseNeuralLayer` with softmax neurons. The number of
         * neurons must equals the number of regressors.
         * @param neuronCount number of neurons to include in the layer
         * @return neural layer containing softmax neurons
         */
        fun softmax(neuronCount: Int): DenseNeuralLayer =
            DenseNeuralLayer((0 until neuronCount).map { index ->
                Neuron(
                    activationFunction = IdentityFunction,
                    regressorCount = neuronCount,
                    includeConstant = false,
                    sampler = UniformSampler(0.0),
                    aggregationFunction = Softmax(index)
                )
            })
    }

    init {
        if (neurons.isEmpty())
            throw RegressionException("A neural layer cannot have 0 neurons")
    }

    constructor(
        numInputs: Int,
        neuronCount: Int,
        activationFunction: FunctionModel,
        includeConstant: Boolean = true,
        sampler: DistributionSampler = NormalSampler(),
        aggregationFunction: AggregationFunction = DotProduct
    ) : this(
        neurons = (0 until neuronCount).map {
            Neuron(
                activationFunction,
                numInputs,
                includeConstant,
                sampler,
                aggregationFunction
            )
        }
    )

    /**
     * Evaluates a vector of inputs and returns the results for each neuron.
     * @param regressors input values
     * @return output values of each neuron
     */
    override fun evaluate(regressors: Vector): Vector =
        Vector(neurons.size) { index ->
            neurons[index].evaluate(regressors)
        }

    /**
     * Returns a copy of the neural layer.
     * @return copy of the neural layer
     */
    override fun copy(): NeuralLayer =
        DenseNeuralLayer(neurons.map { it.copy() })

    /**
     * Copies the weights in `layer` to the weights of this layer's neurons.
     * @param layer layer whose weights should be copied
     */
    override fun updateWeights(layer: NeuralLayer) {
        if (layer !is DenseNeuralLayer)
            throw RegressionException("Cannot update weights to a layer of a different type")

        neurons.forEachIndexed { index, neuron ->
            neuron.updateWeights(layer.neurons[index].weights)
        }
    }

    /**
     * Returns true if `other` is an equivalent `NeuralLayer`.
     * @param other nullable object to compare to this one
     * @return true if other is an equivalent NeuralLayer, false otherwise
     */
    override fun equals(other: Any?): Boolean =
        other is DenseNeuralLayer && numInputs == other.numInputs &&
            neurons == other.neurons
}
