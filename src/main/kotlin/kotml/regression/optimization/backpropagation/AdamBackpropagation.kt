package kotml.regression.optimization.backpropagation

import kotlin.math.pow
import kotlin.math.sqrt
import kotml.extensions.* // ktlint-disable no-wildcard-imports
import kotml.math.MutableVector
import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.cost.CostFunction
import kotml.regression.neural.FeedforwardNeuralNetwork
import kotml.regression.optimization.IterativeOptimizer

class AdamBackpropagation(
    val network: FeedforwardNeuralNetwork,
    val costFunction: CostFunction,
    val betaM: Double = 0.99,
    val betaV: Double = 0.999,
    val epsilon: Double = 0.0001
) : IterativeOptimizer<FeedforwardNeuralNetwork>(
    regressorCount = network.layers.first().neurons.first().weights.coeffs.shape[0],
    outputCount = network.layers.last().neurons.size,
    model = network
) {
    /**
     * Running average of the gradients.
     */
    private val m: List<List<Weights>> = network.layers.map { layer ->
        layer.neurons.map { neuron ->
            val constant: Double? = if (neuron.weights.hasConstant) 0.0 else null
            Weights(constant, MutableVector.zeros(neuron.weights.coeffs.shape[0]))
        }
    }

    /**
     * Stores betaV raised to the power of the number of observations.
     */
    private var betaMProduct: Double = 1.0

    /**
     * Second moments of the gradients.
     */
    private val v: List<List<Weights>> = network.layers.map { layer ->
        layer.neurons.map { neuron ->
            val constant: Double? = if (neuron.weights.hasConstant) 0.0 else null
            Weights(constant, MutableVector.zeros(neuron.weights.coeffs.shape[0]))
        }
    }

    /**
     * Stores betaV raised to the power of the number of observations.
     */
    private var betaVProduct: Double = 1.0

    protected override fun addObservation(regressors: Vector, targets: Vector) {
        betaMProduct *= betaM
        betaVProduct *= betaV

        val inputs = mutableListOf<Vector>()
        val outputs = mutableListOf<Vector>()
        val dIn_dOuts = mutableListOf<List<Vector>>()
        val dIn_dWeights = mutableListOf<List<Weights>>()
        network.layers.forEachIndexed { layerIndex, layer ->
            val input: Vector =
                if (layerIndex == 0)
                    regressors
                else
                    outputs[layerIndex - 1]
            inputs.add(input)
            outputs.add(layer.evaluate(input))
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
                    m[layerIndex][neuronIndex].constant =
                        betaM * m[layerIndex][neuronIndex].constant +
                            (1 - betaM) * new_dErr_dIn[neuronIndex] *
                            dIn_dWeights[layerIndex][neuronIndex].constant
                    v[layerIndex][neuronIndex].constant =
                        betaV * v[layerIndex][neuronIndex].constant +
                            (1 - betaV) * (
                                new_dErr_dIn[neuronIndex] *
                                dIn_dWeights[layerIndex][neuronIndex].constant
                            ).pow(2)
                    val mHat = m[layerIndex][neuronIndex].constant / (1 - betaMProduct)
                    val vHat = v[layerIndex][neuronIndex].constant / (1 - betaVProduct)
                    neuron.weights.constant -= network.stepSize * mHat / (sqrt(vHat) + epsilon)
                }
                inputs[layerIndex].forEachIndexed { coeffIndex, _ ->
                    m[layerIndex][neuronIndex].coeffs[coeffIndex] =
                        betaM * m[layerIndex][neuronIndex].coeffs[coeffIndex] +
                            (1 - betaM) * new_dErr_dIn[neuronIndex] *
                            dIn_dWeights[layerIndex][neuronIndex].coeffs[coeffIndex]
                    v[layerIndex][neuronIndex].coeffs[coeffIndex] =
                        betaV * v[layerIndex][neuronIndex].coeffs[coeffIndex] +
                            (1 - betaV) * (
                                new_dErr_dIn[neuronIndex] *
                                dIn_dWeights[layerIndex][neuronIndex].coeffs[coeffIndex]
                            ).pow(2)
                    val mHat = m[layerIndex][neuronIndex].coeffs[coeffIndex] / (1 - betaMProduct)
                    val vHat = v[layerIndex][neuronIndex].coeffs[coeffIndex] / (1 - betaVProduct)

                    neuron.weights.coeffs[coeffIndex] -= network.stepSize * mHat / (sqrt(vHat) + epsilon)
                }
            }

            dErr_dIn = new_dErr_dIn
        }
    }
}
