package kotml.regression.neural

import kotml.distributions.DistributionSampler
import kotml.distributions.NormalSampler
import kotml.extensions.* // ktlint-disable no-wildcard-imports
import kotml.math.Vector
import kotml.regression.RegressionException
import kotml.regression.cost.CostFunction
import kotml.regression.functions.FunctionOfLinearRegressors

/**
 * `FeedforwardNeuralNetwork` contains a sequence of neural layers that
 * process data in the forward direction (from input layer toward the
 * output layer).
 */
class FeedforwardNeuralNetwork(
    val stepSize: Double,
    val costFunction: CostFunction,
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
        activationFunction: FunctionOfLinearRegressors,
        costFunction: CostFunction,
        includeBias: Boolean = true,
        sampler: DistributionSampler = NormalSampler()
    ) : this(
        stepSize,
        costFunction,
        Array<NeuralLayer>(layerSizes.size) { index ->
            val regressorCount = layerSizes.getOrElse(index - 1) { inputCount }
            NeuralLayer(Array<Neuron>(layerSizes[index]) {
                Neuron(
                    activationFunction = activationFunction,
                    regressorCount = regressorCount,
                    includeBias = includeBias,
                    sampler = sampler
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
}
