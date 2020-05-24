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
 * `NeuralLayer` contains a collection of neurons to be used in a neural
 * network.
 */
class NeuralLayer(val neurons: Array<Neuron>) {
    companion object {
        /**
         * Returns a `NeuralLayer` with softmax neurons. The number of
         * neurons must equals the number of regressors.
         * @param neuronCount number of neurons to include in the layer
         * @return neural layer containing softmax neurons
         */
        fun softmax(neuronCount: Int): NeuralLayer =
            NeuralLayer(Array<Neuron>(neuronCount) { index ->
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
        neuronCount: Int,
        activationFunction: FunctionModel,
        regressorCount: Int,
        includeConstant: Boolean = true,
        sampler: DistributionSampler = NormalSampler(),
        aggregationFunction: AggregationFunction = DotProduct
    ) : this(Array<Neuron>(neuronCount) {
        Neuron(
            activationFunction,
            regressorCount,
            includeConstant,
            sampler,
            aggregationFunction
        )
    })

    /**
     * Evaluates a vector of inputs and returns the results for each neuron.
     * @param regressors input values
     * @return output values of each neuron
     */
    fun evaluate(regressors: Vector): Vector =
        Vector(neurons.size) { index ->
            neurons[index].evaluate(regressors)
        }

    /**
     * Returns a copy of the neural layer.
     * @return copy of the neural layer
     */
    fun copy(): NeuralLayer = NeuralLayer(Array<Neuron>(neurons.size) { index ->
        neurons[index].copy()
    })

    /**
     * Copies the weights in `layer` to the weights of this layer's neurons.
     * @param layer layer whose weights should be copied
     */
    fun updateWeights(layer: NeuralLayer) {
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
        other is NeuralLayer && neurons.contentEquals(other.neurons)
}
