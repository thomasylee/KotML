package kotml.regression.optimization.backpropagation

import kotml.extensions.* // ktlint-disable no-wildcard-imports
import kotml.math.MutableVector
import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.cost.CostFunction
import kotml.regression.neural.FeedforwardNeuralNetwork
import kotml.regression.optimization.IterativeOptimizer

class StochasticBackpropagation(
    val network: FeedforwardNeuralNetwork,
    val costFunction: CostFunction
) : IterativeOptimizer<FeedforwardNeuralNetwork, Vector>(
    regressorCount = network.layers.first().neurons.first().weights.coeffs.shape[0],
    outputCount = network.layers.last().neurons.size,
    model = network
) {
    protected override fun addObservation(regressors: Vector, targets: Vector) {
        observeAndEvaluate(regressors, targets)
    }

    /**
     * Adjusts weights in batches, where each row of regressorsBatch and
     * targetsBatch is an observation of regressors and targets. The
     * network is updated as if the batch weights update at the completion
     * of the batch rather than at the completion of each observation.
     * @param regressorBatch batch of regressors
     * @param targetBatch batch of targets
     */
    override fun batchObserveAndEvaluate(regressorsBatch: Vector, targetsBatch: Vector): Vector {
        val currentNetwork = network.copy()
        return Vector.ofVectors(regressorsBatch.shape[0]) { batchIndex ->
            observeAndEvaluate(
                currentNetwork,
                regressorsBatch(batchIndex),
                targetsBatch(batchIndex)
            )
        }
    }

    override fun observeAndEvaluate(evaluatingModel: FeedforwardNeuralNetwork, regressors: Vector, targets: Vector): Vector {
        val inputs = mutableListOf<Vector>()
        val outputs = mutableListOf<Vector>()
        val dIn_dOuts = mutableListOf<List<Vector>>()
        val dIn_dWeights = mutableListOf<List<Weights>>()
        evaluatingModel.layers.forEachIndexed { layerIndex, layer ->
            val input: Vector =
                if (layerIndex == 0)
                    regressors
                else
                    outputs[layerIndex - 1]
            val output = layer.evaluate(input)
            inputs.add(input)
            outputs.add(output)
            dIn_dOuts.add(layer.neurons.map { neuron ->
                neuron.aggregationFunction.regressorsGradient(neuron.weights, input)
            })
            dIn_dWeights.add(layer.neurons.map { neuron ->
                neuron.aggregationFunction.weightsGradient(neuron.weights, input)
            })
        }

        val costDeriv: Vector = costFunction.gradient(outputs.last(), targets)

        var dErr_dIn: Vector = Vector(0)

        ((network.layers.size - 1) downTo 0).forEach { layerIndex ->
            val layer = network.layers[layerIndex]
            val new_dErr_dIn = MutableVector.zeros(layer.neurons.size)

            layer.neurons.forEachIndexed { neuronIndex, neuron ->
                val dErr_dOut: Double =
                    if (layerIndex == network.layers.size - 1) {
                        costDeriv[neuronIndex]
                    } else {
                        network.layers[layerIndex + 1].neurons.foldIndexed(0.0) { laterNeuronIndex, acc, _ ->
                            acc + dErr_dIn[laterNeuronIndex] *
                                dIn_dOuts[layerIndex + 1][laterNeuronIndex][neuronIndex]
                        }
                    }

                val netInput = neuron.aggregationFunction.aggregate(neuron.weights, inputs[layerIndex])
                val dOut_dIn = neuron.activationFunction.derivative(netInput)
                new_dErr_dIn[neuronIndex] = dErr_dOut * dOut_dIn

                if (neuron.weights.hasConstant) {
                    neuron.weights.constant -= network.stepSize *
                        dErr_dOut *
                        dOut_dIn *
                        dIn_dWeights[layerIndex][neuronIndex].constant
                }
                inputs[layerIndex].forEachIndexed { coeffIndex, _ ->
                    neuron.weights.coeffs[coeffIndex] -= network.stepSize *
                        dErr_dOut *
                        dOut_dIn *
                        dIn_dWeights[layerIndex][neuronIndex].coeffs[coeffIndex]
                }
            }

            dErr_dIn = new_dErr_dIn
        }

        return outputs.last()
    }
}
