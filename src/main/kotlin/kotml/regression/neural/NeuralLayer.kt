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
        fun softmax(neuronCount: Int, regressorCount: Int): NeuralLayer =
            NeuralLayer(Array<Neuron>(neuronCount) { index ->
                Neuron(
                    activationFunction = IdentityFunction,
                    regressorCount = regressorCount,
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
}
