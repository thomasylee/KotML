package kotml.regression.linear

import kotml.math.Matrix
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class StochasticGradientDescentTest {
    @Test
    fun `calculates weights correctly`() {
        val estimator = StochasticGradientDescent(0.001, { x -> x })
        estimator.addObservation(-19.0, Matrix(-9.0))
        estimator.addObservation(-17.0, Matrix(-8.0))
        estimator.addObservation(-15.0, Matrix(-7.0))
        assertEquals(-0.088344472, estimator.weights[0])
        assertEquals(0.722863304, estimator.weights[1])
        assertEquals(3.525972048, estimator.estimate(Matrix(5.0)))
    }
}
