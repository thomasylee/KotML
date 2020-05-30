package kotml.regression.optimization.backpropagation

import kotlin.math.pow
import kotlin.math.sqrt
import kotml.math.MutableVector
import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.cost.CostFunction
import kotml.regression.neural.DenseNeuralLayer
import kotml.regression.neural.FeedforwardNeuralNetwork

/**
 * `AdamBackpropagation` performs Adam-optimized backpropagation on a neural
 * network, with optional weight decay hyperparameters available to use
 * AdamW instead of classic Adam.
 *
 * References:
 * * Decoupled Weight Decay Regulatization (2019) - Ilya Loshchilov, Frank
 *   Hutter - https://arxiv.org/abs/1711.05101
 */
class AdamBackpropagation(
    network: FeedforwardNeuralNetwork,
    costFunction: CostFunction,
    val stepSize: Double,
    val weightDecayRate: Double = 0.0,
    val weightDecayScalingFactor: Double = 1.0,
    val betaM: Double = 0.99,
    val betaV: Double = 0.999,
    val epsilon: Double = 0.0001
) : Backpropagation(
    network = network,
    costFunction = costFunction
) {
    /**
     * Running average of the gradients. The first index is the dense layer
     * index, and the second index is the neuron index.
     */
    private val m: List<List<Weights>> = model.denseLayers.map { layer ->
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
     * Second moments of the gradients. The first index is the dense layer
     * index, and the second index is the neuron index.
     */
    private val v: List<List<Weights>> = model.denseLayers.map { layer ->
        layer.neurons.map { neuron ->
            val constant: Double? = if (neuron.weights.hasConstant) 0.0 else null
            Weights(constant, MutableVector.zeros(neuron.weights.coeffs.shape[0]))
        }
    }

    override fun observeAndEvaluate(
        evaluatingModel: FeedforwardNeuralNetwork,
        regressors: Vector,
        targets: Vector
    ): Vector {
        betaMProduct *= betaM
        betaVProduct *= betaV
        return super.observeAndEvaluate(evaluatingModel, regressors, targets)
    }

    /**
     * Stores betaV raised to the power of the number of observations.
     */
    private var betaVProduct: Double = 1.0

    protected override fun backpropagateLayer(
        layer: DenseNeuralLayer,
        denseLayerIndex: Int,
        dErr_dIns: Map<DenseNeuralLayer, Vector>,
        dIn_dWeights: Map<DenseNeuralLayer, List<Weights>>
    ) {
        val dErr_dIn = dErr_dIns.getValue(layer)

        model.denseLayers[denseLayerIndex].neurons.forEachIndexed { neuronIndex, neuron ->
            if (neuron.weights.hasConstant) {
                m[denseLayerIndex][neuronIndex].constant =
                    betaM * m[denseLayerIndex][neuronIndex].constant +
                        (1 - betaM) * dErr_dIn[neuronIndex] *
                        dIn_dWeights.getValue(layer)[neuronIndex].constant
                v[denseLayerIndex][neuronIndex].constant =
                    betaV * v[denseLayerIndex][neuronIndex].constant +
                        (1 - betaV) * (
                            dErr_dIn[neuronIndex] *
                            dIn_dWeights.getValue(layer)[neuronIndex].constant
                        ).pow(2)
                val mHat = m[denseLayerIndex][neuronIndex].constant / (1 - betaMProduct)
                val vHat = v[denseLayerIndex][neuronIndex].constant / (1 - betaVProduct)
                neuron.weights.constant -= weightDecayScalingFactor * (
                    stepSize * mHat / (sqrt(vHat) + epsilon) +
                        weightDecayRate * neuron.weights.constant
                )
            }
            (0 until layer.numInputs).forEach { coeffIndex ->
                m[denseLayerIndex][neuronIndex].coeffs[coeffIndex] =
                    betaM * m[denseLayerIndex][neuronIndex].coeffs[coeffIndex] +
                        (1 - betaM) * dErr_dIn[neuronIndex] *
                        dIn_dWeights.getValue(layer)[neuronIndex].coeffs[coeffIndex]
                v[denseLayerIndex][neuronIndex].coeffs[coeffIndex] =
                    betaV * v[denseLayerIndex][neuronIndex].coeffs[coeffIndex] +
                        (1 - betaV) * (
                            dErr_dIn[neuronIndex] *
                            dIn_dWeights.getValue(layer)[neuronIndex].coeffs[coeffIndex]
                        ).pow(2)
                val mHat = m[denseLayerIndex][neuronIndex].coeffs[coeffIndex] / (1 - betaMProduct)
                val vHat = v[denseLayerIndex][neuronIndex].coeffs[coeffIndex] / (1 - betaVProduct)

                neuron.weights.coeffs[coeffIndex] -= weightDecayScalingFactor * (
                    stepSize * mHat / (sqrt(vHat) + epsilon) +
                        weightDecayRate * neuron.weights.coeffs[coeffIndex]
                )
            }
        }
    }
}
