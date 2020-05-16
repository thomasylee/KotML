package kotml.regression.neural

import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.functions.IdentityFunction
import kotml.regression.functions.ReLU
import kotml.regression.functions.aggregation.Softmax
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class NeuralLayerTest {
    @Test
    fun `evaluate() returns the correct value`() {
        val neuralLayer = NeuralLayer(Array<Neuron>(3) { index ->
            Neuron(
                activationFunction = ReLU,
                weights = Weights(index.toDouble(), Vector(2.0))
            )
        })
        assertEquals(Vector(4, 5, 6), neuralLayer.evaluate(Vector(2)))
    }

    @Test
    fun `softmax() returns a softmax neural layer`() {
        val neuralLayer = NeuralLayer.softmax(5, 3)

        assertEquals(5, neuralLayer.neurons.size)
        neuralLayer.neurons.forEachIndexed { index, neuron ->
            assertTrue(neuron.aggregationFunction is Softmax)
            assertEquals(index, (neuron.aggregationFunction as Softmax).regressorIndex)
            assertFalse(neuron.weights.hasConstant)
            assertTrue(neuron.activationFunction is IdentityFunction)
        }
    }
}
