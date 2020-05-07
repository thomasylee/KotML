package kotml.regression.neural

import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.functions.ReLU
import kotml.regression.objectives.OrdinaryLeastSquares
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class FeedforwardNeuralNetworkTest {
    @Test
    fun `evaluate() returns the correct output values`() {
        val hiddenLayer = NeuralLayer(Array<Neuron>(2) { index ->
            Neuron(
                stepSize = 0.1,
                activationFunction = ReLU,
                objectiveFunction = OrdinaryLeastSquares,
                weights = Weights(index.toDouble(), doubleArrayOf(-0.25))
            )
        })
        val outputLayer = NeuralLayer(Array<Neuron>(2) { index ->
            Neuron(
                stepSize = 0.1,
                activationFunction = ReLU,
                objectiveFunction = OrdinaryLeastSquares,
                weights = Weights(index.toDouble(), doubleArrayOf(2.0, -0.5))
            )
        })
        // Input: [2]
        // Hidden layer: [0 - 0.25 * 2 => 0, 1 - 0.25 * 2 => 0.5]
        // Output layer: [0 + 2*0 - 0.5*0.5 => 0, 1 + 2*0 - 0.5*0.5 => 0.75]
        val network = FeedforwardNeuralNetwork(
            stepSize = 0.1,
            layers = arrayOf(hiddenLayer, outputLayer)
        )
        assertEquals(Vector(0.0, 0.75), network.evaluate(Vector(2)))
    }
}
