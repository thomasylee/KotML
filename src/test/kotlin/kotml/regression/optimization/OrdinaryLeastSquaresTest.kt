package kotml.regression.optimization

import kotml.math.Vector
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Test

class OrdinaryLeastSquaresTest {
    @Test
    fun `processBatch() calculates weights correctly`() {
        val estimator = OrdinaryLeastSquares(2)
        assertFalse(estimator.weights.hasConstant)
        assertEquals(2, estimator.weights.coeffs.shape[0])
        // f(x1, x2) = 2 * x1 - x2
        estimator.observe(Vector(0, 2), Vector(-2))
        estimator.observe(Vector(1, 3), Vector(-1))
        estimator.observe(Vector(2, 4), Vector(0))
        estimator.processBatch()
        assertEquals(2.0, estimator.weights.coeffs[0])
        assertEquals(-1.0, estimator.weights.coeffs[1])
    }
}
