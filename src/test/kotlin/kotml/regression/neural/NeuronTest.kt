package kotml.regression.neural

import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.functions.ReLU
import kotml.regression.objectives.OrdinaryLeastSquares
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class NeuronTest {
    @Test
    fun `evaluate() returns the correct value`() {
        val neuron = Neuron(
            stepSize = 0.1,
            activationFunction = ReLU,
            objectiveFunction = OrdinaryLeastSquares,
            weights = Weights(1.0, doubleArrayOf(2.0))
        )
        assertEquals(5.0, neuron.evaluate(Vector(2)))
    }
}
