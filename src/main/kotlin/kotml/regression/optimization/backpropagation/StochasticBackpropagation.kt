package kotml.regression.optimization.backpropagation

import kotml.extensions.* // ktlint-disable no-wildcard-imports
import kotml.math.MutableVector
import kotml.math.Vector
import kotml.regression.cost.CostFunction
import kotml.regression.functions.LinearFunction
import kotml.regression.neural.FeedforwardNeuralNetwork
import kotml.regression.optimization.IterativeOptimizer

class StochasticBackpropagation(
    val network: FeedforwardNeuralNetwork,
    val costFunction: CostFunction
) : IterativeOptimizer<FeedforwardNeuralNetwork>(
    regressorCount = network.layers.first().neurons.first().weights.coeffs.shape[0],
    outputCount = network.layers.last().neurons.size,
    model = network
) {
    protected override fun addObservation(regressors: Vector, targets: Vector) {
        val inputs = mutableListOf<Vector>()
        val outputs = mutableListOf<Vector>()
        network.layers.forEachIndexed { layerIndex, layer ->
            val input: Vector =
                if (layerIndex == 0)
                    regressors
                else
                    outputs[layerIndex - 1]
            inputs.add(input)
            outputs.add(layer.evaluate(input))
        }

        val costDeriv: Vector = costFunction.gradient(outputs.last(), targets)

        var dErr_dNetIn: Vector = Vector(0)

        ((network.layers.size - 1) downTo 0).forEach { layerIndex ->
            val layer = network.layers[layerIndex]
            val new_dErr_dNetIn = MutableVector.zeros(layer.neurons.size)

            layer.neurons.forEachIndexed { neuronIndex, neuron ->
                val netInput = LinearFunction.evaluate(neuron.weights, inputs[layerIndex])
                val dErr_dOut: Double =
                    if (layerIndex == network.layers.size - 1) {
                        costDeriv[neuronIndex]
                    } else {
                        network.layers[layerIndex + 1].neurons.foldIndexed(0.0) { laterNeuronIndex, acc, laterNeuron ->
                            acc + dErr_dNetIn[laterNeuronIndex] * laterNeuron.weights.coeffs[neuronIndex]
                        }
                    }
                val dOut_dNetIn = neuron.activationFunction.netInputGradient(netInput)
                new_dErr_dNetIn[neuronIndex] = dErr_dOut * dOut_dNetIn

                if (neuron.weights.hasConstant) {
                    neuron.weights.constant -= network.stepSize * new_dErr_dNetIn[neuronIndex]
                }
                inputs[layerIndex].forEachIndexed { coeffIndex, input ->
                    neuron.weights.coeffs[coeffIndex] = neuron.weights.coeffs[coeffIndex] -
                        network.stepSize * new_dErr_dNetIn[neuronIndex] * input
                }
            }

            dErr_dNetIn = new_dErr_dNetIn
        }
    }
}
