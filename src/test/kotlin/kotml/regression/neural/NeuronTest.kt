package kotml.regression.neural

import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.functions.ReLU
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotEquals
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

    @Test
    fun `copy() returns a copy of the neuron`() {
        val original = Neuron(ReLU, Weights(Vector(1)))
        val copy = original.copy()

        assertEquals(original, copy)
        original.weights.coeffs[0] += 1.0
        assertNotEquals(original, copy)
    }

    @Test
    fun `updateWeights() copies the weights`() {
        val neuron = Neuron(ReLU, Weights(2.0, Vector(3)))
        val newWeights = Weights(5.0, Vector(4))

        neuron.updateWeights(newWeights)
        assertEquals(newWeights, neuron.weights)

        newWeights.constant = 10.0
        newWeights.coeffs[0] = 20.0
        assertEquals(Weights(10.0, Vector(20)), newWeights)
        assertEquals(Weights(5.0, Vector(4)), neuron.weights)
    }
}
