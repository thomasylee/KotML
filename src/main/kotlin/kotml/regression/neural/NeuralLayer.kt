package kotml.regression.neural

import kotml.math.Vector
import kotml.regression.optimization.WeightedOptimizer

/**
 * `NeuralLayer` contains a collection of neurons to be used in a neural
 * network.
 */
class NeuralLayer(val neurons: Array<Neuron>) {
    constructor(optimizer: WeightedOptimizer, size: Int) : this(Array<Neuron>(size) {
        Neuron(optimizer)
    })

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
