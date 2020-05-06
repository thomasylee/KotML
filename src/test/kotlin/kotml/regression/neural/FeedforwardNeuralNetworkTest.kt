package kotml.regression.neural

import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.functions.Polynomial
import kotml.regression.objectives.OrdinaryLeastSquares
import kotml.regression.optimization.StochasticGradientDescent
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class FeedforwardNeuralNetworkTest {
    @Test
    fun `evaluate() returns the correct output values`() {
        val hiddenLayer = NeuralLayer(Array<Neuron>(2) { index ->
            Neuron(StochasticGradientDescent(
                stepSize = 0.1,
                function = Polynomial(Vector(2)),
                costFunction = OrdinaryLeastSquares,
                weights = Weights(index.toDouble(), doubleArrayOf(-1.0))
            ))
        })
        val outputLayer = NeuralLayer(Array<Neuron>(2) { index ->
            Neuron(StochasticGradientDescent(
                stepSize = 0.1,
                function = Polynomial(Vector(1, 2)),
                costFunction = OrdinaryLeastSquares,
                weights = Weights(index.toDouble(), doubleArrayOf(2.0, 0.5))
            ))
        })
        // Hidden layer: [-4, -3]
        // Output layer: [0+2(-4)+0.5(-3)^2, 1+2(-4)+0.5(-3)^2] = [-3.5, -2.5]
        val network = FeedforwardNeuralNetwork(arrayOf(hiddenLayer, outputLayer))
        assertEquals(Vector(-3.5, -2.5), network.evaluate(Vector(2.0)))
    }
}
