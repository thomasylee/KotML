package kotml.regression.optimization

import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.cost.loss.OrdinaryLeastSquares
import kotml.regression.functions.Polynomial
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class StochasticGradientDescentTest {
    @Test
    fun `calculates weights correctly with bias`() {
        val estimator = StochasticGradientDescent(
            0.001, Polynomial(Vector(1.0)), OrdinaryLeastSquares, 1)
        assertTrue(estimator.weights.hasBias)
        assertEquals(1, estimator.weights.coeffs.shape[0])
        estimator.addObservation(-19.0, Vector(-9.0))
        estimator.addObservation(-17.0, Vector(-8.0))
        estimator.addObservation(-15.0, Vector(-7.0))
        assertEquals(-0.088344472, estimator.weights.bias)
        assertEquals(0.722863304, estimator.weights.coeffs[0])
        assertEquals(3.525972048, estimator.function.evaluate(estimator.weights, Vector(5.0)))
    }

    @Test
    fun `calculates weights correctly without bias`() {
        val estimator = StochasticGradientDescent(
            0.02, Polynomial(Vector(1.0)), OrdinaryLeastSquares, 1, false)
        estimator.addObservation(0.0, Vector(0.0))
        estimator.addObservation(2.0, Vector(1.0))
        estimator.addObservation(4.0, Vector(2.0))
        estimator.addObservation(6.0, Vector(3.0))
        estimator.addObservation(8.0, Vector(4.0))
        estimator.addObservation(10.0, Vector(5.0))
        assertFalse(estimator.weights.hasBias)
        assertEquals(1, estimator.weights.coeffs.shape[0])
        assertEquals(2.0, estimator.weights.coeffs[0])
        assertEquals(12.0, estimator.function.evaluate(estimator.weights, Vector(6.0)))
    }

    @Test
    fun `providing initWeights sets the weights correctly`() {
        val estimator = StochasticGradientDescent(
            stepSize = 0.001,
            function = Polynomial(Vector(1.0)),
            lossFunction = OrdinaryLeastSquares,
            weights = Weights(-0.038, Vector(0.342))
        )
        estimator.addObservation(-17.0, Vector(-8.0))
        estimator.addObservation(-15.0, Vector(-7.0))
        assertEquals(-0.088344472, estimator.weights.bias)
        assertEquals(0.722863304, estimator.weights.coeffs[0])
        assertEquals(3.525972048, estimator.function.evaluate(estimator.weights, Vector(5.0)))
    }
}
