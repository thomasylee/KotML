package kotml.regression.neural

import kotml.distributions.DistributionSampler
import kotml.distributions.NormalSampler
import kotml.math.Vector
import kotml.regression.RegressionException
import kotml.regression.functions.FunctionOfLinearRegressors

/**
 * `NeuralLayer` contains a collection of neurons to be used in a neural
 * network.
 */
class NeuralLayer(val neurons: Array<Neuron>) {
    constructor(
        neuronCount: Int,
        activationFunction: FunctionOfLinearRegressors,
        regressorCount: Int,
        includeBias: Boolean = true,
        sampler: DistributionSampler = NormalSampler()
    ) : this(Array<Neuron>(neuronCount) {
        Neuron(
            activationFunction,
            regressorCount,
            includeBias,
            sampler
        )
    })

    init {
        if (neurons.isEmpty())
            throw RegressionException("A neural layer cannot have 0 neurons")
    }

    /**
     * Evaluates a vector of inputs and returns the results for each neuron.
     * @param regressors input values
     * @return output values of each neuron
     */
    fun evaluate(regressors: Vector): Vector =
        Vector(*DoubleArray(neurons.size) { index ->
            neurons[index].evaluate(regressors)
        })
}
