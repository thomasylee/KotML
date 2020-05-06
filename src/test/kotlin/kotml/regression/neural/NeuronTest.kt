package kotml.regression.neural

import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.functions.Polynomial
import kotml.regression.objectives.OrdinaryLeastSquares
import kotml.regression.optimization.StochasticGradientDescent
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class NeuronTest {
    @Test
    fun `evaluate() returns the correct value`() {
        val neuron = Neuron(StochasticGradientDescent(
            stepSize = 0.1,
            function = Polynomial(Vector(2.0)),
            costFunction = OrdinaryLeastSquares,
            weights = Weights(1.0, doubleArrayOf(2.0))
        ))
        assertEquals(9.0, neuron.evaluate(Vector(2.0)))
    }
}
