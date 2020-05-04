package kotml.regression.objectives

import kotml.math.Vector
import kotml.regression.functions.Polynomial
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class OrdinaryLeastSquaresTest {
    @Test
    fun `evaluate() returns the correct cost with bias`() {
        assertEquals(9.0, OrdinaryLeastSquares.evaluate(
            function = Polynomial(Vector(2)),
            weights = doubleArrayOf(1.0, 2.0),
            regressors = Vector(3),
            response = 16.0))
    }

    @Test
    fun `evaluate() returns the correct cost without bias`() {
        assertEquals(4.0, OrdinaryLeastSquares.evaluate(
            function = Polynomial(Vector(2)),
            weights = doubleArrayOf(2.0),
            regressors = Vector(3),
            response = 16.0))
    }
}
