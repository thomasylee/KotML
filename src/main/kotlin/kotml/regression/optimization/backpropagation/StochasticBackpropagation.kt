package kotml.regression.optimization.backpropagation

import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.cost.CostFunction
import kotml.regression.neural.DenseNeuralLayer
import kotml.regression.neural.FeedforwardNeuralNetwork

class StochasticBackpropagation(
    network: FeedforwardNeuralNetwork,
    costFunction: CostFunction,
    val stepSize: Double,
    val weightDecayRate: Double = 0.0,
    val weightDecayScalingFactor: Double = 1.0
) : Backpropagation(
    network = network,
    costFunction = costFunction
) {
    protected override fun backpropagateLayer(
        layer: DenseNeuralLayer,
        denseLayerIndex: Int,
        dErr_dIns: Map<DenseNeuralLayer, Vector>,
        dIn_dWeights: Map<DenseNeuralLayer, List<Weights>>
    ) {
        val dErr_dIn = dErr_dIns.getValue(layer)

        model.denseLayers[denseLayerIndex].neurons.forEachIndexed { neuronIndex, neuron ->
            if (neuron.weights.hasConstant) {
                neuron.weights.constant -= stepSize *
                    dErr_dIn[neuronIndex] *
                    dIn_dWeights.getValue(layer)[neuronIndex].constant +
                    // Weight decay
                    weightDecayScalingFactor * weightDecayRate * neuron.weights.constant
            }
            (0 until layer.numInputs).forEach { coeffIndex ->
                neuron.weights.coeffs[coeffIndex] -= stepSize *
                    dErr_dIn[neuronIndex] *
                    dIn_dWeights.getValue(layer)[neuronIndex].coeffs[coeffIndex] +
                    // Weight decay
                    weightDecayScalingFactor * weightDecayRate * neuron.weights.coeffs[coeffIndex]
            }
        }
    }
}
