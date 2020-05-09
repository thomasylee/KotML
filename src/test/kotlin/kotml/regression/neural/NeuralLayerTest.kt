package kotml.regression.neural

import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.functions.ReLU
import org.junit.jupiter.api.Assertions.assertEquals
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
}
