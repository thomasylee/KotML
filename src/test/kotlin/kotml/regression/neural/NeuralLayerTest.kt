package kotml.regression.neural

import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.functions.Polynomial
import kotml.regression.objectives.OrdinaryLeastSquares
import kotml.regression.optimization.StochasticGradientDescent
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class NeuralLayerTest {
    @Test
    fun `evaluate() returns the correct value`() {
        val neuralLayer = NeuralLayer(Array<Neuron>(3) { index ->
            Neuron(StochasticGradientDescent(
                stepSize = 0.1,
                function = Polynomial(Vector(2.0)),
                costFunction = OrdinaryLeastSquares,
                weights = Weights(index.toDouble(), doubleArrayOf(2.0))
            ))
        })
        assertEquals(Vector(8, 9, 10), neuralLayer.evaluate(Vector(2.0)))
    }
}
