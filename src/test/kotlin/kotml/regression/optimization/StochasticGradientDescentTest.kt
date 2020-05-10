package kotml.regression.optimization

import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.cost.loss.SquaredError
import kotml.regression.functions.Polynomial
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class StochasticGradientDescentTest {
    @Test
    fun `calculates weights correctly with a constant`() {
        val estimator = StochasticGradientDescent(
            0.001, Polynomial(Vector(1)), SquaredError, 1)
        assertTrue(estimator.weights.hasConstant)
        assertEquals(1, estimator.weights.coeffs.shape[0])
        estimator.addObservation(Vector(-9), Vector(-19))
        estimator.addObservation(Vector(-8), Vector(-17))
        estimator.addObservation(Vector(-7), Vector(-15))
        assertEquals(-0.088344472, estimator.weights.constant)
        assertEquals(0.722863304, estimator.weights.coeffs[0])
        assertEquals(3.525972048, estimator.function.evaluate(estimator.weights, Vector(5.0)))
    }

    @Test
    fun `calculates weights correctly without a constant`() {
        val estimator = StochasticGradientDescent(
            0.02, Polynomial(Vector(1)), SquaredError, 1, false)
        estimator.addObservation(Vector(0.0), Vector(0))
        estimator.addObservation(Vector(1), Vector(2))
        estimator.addObservation(Vector(2), Vector(4))
        estimator.addObservation(Vector(3), Vector(6))
        estimator.addObservation(Vector(4), Vector(8))
        estimator.addObservation(Vector(5), Vector(10))
        assertFalse(estimator.weights.hasConstant)
        assertEquals(1, estimator.weights.coeffs.shape[0])
        assertEquals(2.0, estimator.weights.coeffs[0])
        assertEquals(12.0, estimator.function.evaluate(estimator.weights, Vector(6.0)))
    }

    @Test
    fun `providing initWeights sets the weights correctly`() {
        val estimator = StochasticGradientDescent(
            stepSize = 0.001,
            function = Polynomial(Vector(1)),
            lossFunction = SquaredError,
            weights = Weights(-0.038, Vector(0.342))
        )
        estimator.addObservation(Vector(-8), Vector(-17))
        estimator.addObservation(Vector(-7), Vector(-15))
        assertEquals(-0.088344472, estimator.weights.constant)
        assertEquals(0.722863304, estimator.weights.coeffs[0])
        assertEquals(3.525972048, estimator.function.evaluate(estimator.weights, Vector(5.0)))
    }
}
