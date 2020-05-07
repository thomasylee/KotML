package kotml.regression.neural

import kotml.distributions.DistributionSampler
import kotml.distributions.NormalSampler
import kotml.math.Vector
import kotml.regression.functions.FunctionOfLinearRegressors
import kotml.regression.objectives.ObjectiveFunction

/**
 * `NeuralLayer` contains a collection of neurons to be used in a neural
 * network.
 */
class NeuralLayer(val neurons: Array<Neuron>) {
    constructor(
        stepSize: Double,
        neuronCount: Int,
        activationFunction: FunctionOfLinearRegressors,
        objectiveFunction: ObjectiveFunction,
        regressorCount: Int,
        includeBias: Boolean = true,
        sampler: DistributionSampler = NormalSampler()
    ) : this(Array<Neuron>(neuronCount) {
        Neuron(
            stepSize,
            activationFunction,
            objectiveFunction,
            regressorCount,
            includeBias,
            sampler
        )
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
