package kotml.regression.optimization

import kotml.math.Vector
import kotml.regression.functions.Polynomial
import kotml.regression.objectives.OrdinaryLeastSquares
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class StochasticGradientDescentTest {
    @Test
    fun `calculates weights correctly`() {
        val estimator = StochasticGradientDescent(
            0.001, 1, Polynomial(Vector(1.0)), OrdinaryLeastSquares)
        estimator.addObservation(-19.0, Vector(-9.0))
        estimator.addObservation(-17.0, Vector(-8.0))
        estimator.addObservation(-15.0, Vector(-7.0))
        assertEquals(-0.088344472, estimator.weights[0])
        assertEquals(0.722863304, estimator.weights[1])
        assertEquals(3.525972048, estimator.function.evaluate(estimator.weights, Vector(5.0)))
    }

    @Test
    fun `providing initWeights sets the weights correctly`() {
        val estimator = StochasticGradientDescent(
            stepSize = 0.001,
            regressorCount = 1,
            function = Polynomial(Vector(1.0)),
            costFunction = OrdinaryLeastSquares,
            initWeights = doubleArrayOf(-0.038, 0.342)
        )
        estimator.addObservation(-17.0, Vector(-8.0))
        estimator.addObservation(-15.0, Vector(-7.0))
        assertEquals(-0.088344472, estimator.weights[0])
        assertEquals(0.722863304, estimator.weights[1])
        assertEquals(3.525972048, estimator.function.evaluate(estimator.weights, Vector(5.0)))
    }
}
