package kotml.regression.neural

import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.functions.ReLU
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class NeuronTest {
    @Test
    fun `evaluate() returns the correct value`() {
        val neuron = Neuron(
            activationFunction = ReLU,
            weights = Weights(1.0, Vector(2.0))
        )
        assertEquals(5.0, neuron.evaluate(Vector(2)))
    }
}
