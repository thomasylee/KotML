package kotml.regression.optimization.backpropagation

import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.cost.CostFunction
import kotml.regression.neural.DenseNeuralLayer
import kotml.regression.neural.FeedforwardNeuralNetwork
import kotml.regression.optimization.IterativeOptimizer

/**
 * `Backpropagation` takes care of most of the complexity around evaluating
 * and determining gradients for neural network layers (including split
 * layers), so subclasses can focus on weight update logic.
 */
abstract class Backpropagation(
    val network: FeedforwardNeuralNetwork,
    val costFunction: CostFunction
) : IterativeOptimizer<FeedforwardNeuralNetwork, Vector>(
    regressorCount = network.layers.first().numInputs,
    outputCount = network.layers.last().numOutputs,
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
     * @param regressorsBatch batch of regressors, shape = (batchSize, numRegressors)
     * @param targetsBatch batch of targets, shape = (batchSize, numTargets)
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

    private fun calculateGradients(
        evaluatingModel: FeedforwardNeuralNetwork,
        layer: DenseNeuralLayer,
        regressors: Vector,
        inputs: MutableMap<DenseNeuralLayer, Vector>,
        outputs: MutableMap<DenseNeuralLayer, Vector>,
        dIn_dOuts: MutableMap<DenseNeuralLayer, List<Vector>>,
        dIn_dWeights: MutableMap<DenseNeuralLayer, List<Weights>>
    ): List<DenseNeuralLayer> {
        val prevLayers = evaluatingModel.prevDenseLayers[layer]
        val input =
            if (prevLayers.isNullOrEmpty())
                regressors
            else
                Vector.ofVectors(prevLayers.size) { index ->
                    outputs.getValue(prevLayers[index])
                }.flatten()

        inputs.put(layer, input)
        outputs.put(layer, layer.evaluate(input))
        dIn_dOuts.put(layer, layer.neurons.map { neuron ->
            neuron.aggregationFunction.regressorsGradient(neuron.weights, input)
        })
        dIn_dWeights.put(layer, layer.neurons.map { neuron ->
            neuron.aggregationFunction.weightsGradient(neuron.weights, input)
        })

        return listOf(layer)
    }

    /**
     * Returns the output of the last layers of `evaluatingModel`, which may
     * not be the primary model. If `evaluatingModel` is a different model,
     * then the weights of `evaluatingModel` remain unchanged and are only
     * used to evaluate values while the primary model's weights are udpated.
     * @param evaluatingModel model used to evaluate the regressors
     * @param regressors independent variable values
     * @param targets dependent variable values
     * @return evaluated output of the evaluatingModel
     */
    override fun observeAndEvaluate(evaluatingModel: FeedforwardNeuralNetwork, regressors: Vector, targets: Vector): Vector {
        // The keys to these maps are the dense layers of the evaluatingModel,
        // not the primary model.
        val inputs = mutableMapOf<DenseNeuralLayer, Vector>()
        val outputs = mutableMapOf<DenseNeuralLayer, Vector>()
        val dIn_dOuts = mutableMapOf<DenseNeuralLayer, List<Vector>>()
        val dIn_dWeights = mutableMapOf<DenseNeuralLayer, List<Weights>>()

        evaluatingModel.denseLayers.forEach { layer ->
            calculateGradients(
                evaluatingModel, layer, regressors, inputs, outputs,
                dIn_dOuts, dIn_dWeights
            )
        }
        val lastLayers = evaluatingModel.lastDenseLayers

        val lastOutput = Vector.ofVectors(lastLayers.size) { index ->
            outputs.getValue(lastLayers[index])
        }.flatten()

        val costDeriv: Vector = costFunction.gradient(lastOutput, targets)
        val dErr_dIns = mutableMapOf<DenseNeuralLayer, Vector>()

        lastLayers.forEach { layer ->
            val dErr_dIn = Vector(layer.neurons.size) { neuronIndex ->
                val neuron = layer.neurons[neuronIndex]
                val dErr_dOut = costDeriv[neuronIndex]
                val netInput = neuron.aggregationFunction.aggregate(
                    neuron.weights, inputs.getValue(layer))
                val dOut_dIn = neuron.activationFunction.derivative(netInput)
                dErr_dOut * dOut_dIn
            }
            dErr_dIns.put(layer, dErr_dIn)
        }

        // Backpropagation must be breadth-first rather than depth-first
        // to ensure that dErr_dIn for later layers have already been
        // calculated.
        val denseLayers = evaluatingModel.denseLayers
        ((denseLayers.size - 1) downTo 0).forEach { layerIndex ->
            val layer = denseLayers[layerIndex]

            backpropagateLayer(layer, layerIndex, dErr_dIns, dIn_dWeights)

            val prevLayers = evaluatingModel.prevDenseLayers[layer]
            if (!prevLayers.isNullOrEmpty()) {
                prevLayers.forEach { prevLayer ->
                    val new_dErr_dIn = Vector(prevLayer.neurons.size) { neuronIndex ->
                        val neuron = prevLayer.neurons[neuronIndex]
                        val dErr_dOut = evaluatingModel.nextDenseLayers.getValue(prevLayer).fold(0.0) { totalAcc, nextLayer ->
                            nextLayer.neurons.foldIndexed(totalAcc) { laterNeuronIndex, acc, _ ->
                                acc + dErr_dIns.getValue(nextLayer)[laterNeuronIndex] *
                                    dIn_dOuts.getValue(nextLayer)[laterNeuronIndex][neuronIndex]
                            }
                        }
                        val netInput = neuron.aggregationFunction.aggregate(neuron.weights, inputs.getValue(prevLayer))
                        val dOut_dIn = neuron.activationFunction.derivative(netInput)
                        dErr_dOut * dOut_dIn
                    }
                    dErr_dIns.put(prevLayer, new_dErr_dIn)
                }
            }
        }

        return lastOutput
    }

    /**
     * Applies weight updates according to the backpropagation algorithm.
     * @param layer current layer being evaluated for backprogation, which
     *   may be from a different model (evaluatingModel) than the main model
     * @param denseLayerIndex index of the dense layer being updated
     * @param dErr_dIns gradients of error wrt inputs, which is the key
     *   gradient to carry backward in backpropagation
     * @param dIn_dWeights gradients of input with respect to the weights
     */
    protected abstract fun backpropagateLayer(
        layer: DenseNeuralLayer,
        denseLayerIndex: Int,
        dErr_dIns: Map<DenseNeuralLayer, Vector>,
        dIn_dWeights: Map<DenseNeuralLayer, List<Weights>>
    )
}
