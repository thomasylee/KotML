package kotml.regression.neural

import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.functions.ReLU
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotEquals
import org.junit.jupiter.api.Test

class FeedforwardNeuralNetworkTest {
    @Test
    fun `evaluate() returns the correct output values`() {
        val hiddenLayer = NeuralLayer(Array<Neuron>(2) { index ->
            Neuron(
                activationFunction = ReLU,
                weights = Weights(index.toDouble(), Vector(-0.25))
            )
        })
        val outputLayer = NeuralLayer(Array<Neuron>(2) { index ->
            Neuron(
                activationFunction = ReLU,
                weights = Weights(index.toDouble(), Vector(2.0, -0.5))
            )
        })
        val network = FeedforwardNeuralNetwork(
            stepSize = 0.1,
            layers = arrayOf(hiddenLayer, outputLayer)
        )
        // Input: [2]
        // Hidden layer: [0 - 0.25 * 2 => 0, 1 - 0.25 * 2 => 0.5]
        // Output layer: [0 + 2*0 - 0.5*0.5 => 0, 1 + 2*0 - 0.5*0.5 => 0.75]
        assertEquals(Vector(0.0, 0.75), network.evaluate(Vector(2)))
    }

    @Test
    fun `copy() returns a copy of the neural network`() {
        val original = FeedforwardNeuralNetwork(0.1, 1, intArrayOf(1), ReLU)
        var copy = original.copy()
        assertEquals(original, copy)
        original.layers.first().neurons.first().weights.coeffs[0] += 1.0
        assertNotEquals(original, copy)

        copy = original.copy()
        assertEquals(original, copy)
        original.layers.first().neurons[0] = Neuron(ReLU, Weights(Vector(5)))
        assertNotEquals(original, copy)

        copy = original.copy()
        assertEquals(original, copy)
        original.layers[0] = NeuralLayer(arrayOf(Neuron(ReLU, Weights(Vector(10)))))
        assertNotEquals(original, copy)
    }
}
