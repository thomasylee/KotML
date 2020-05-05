package kotml.regression.objectives

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class OrdinaryLeastSquaresTest {
    @Test
    fun `evaluate() returns the correct cost`() {
        assertEquals(4.0, OrdinaryLeastSquares.evaluate(3.0, 5.0))
    }

    @Test
    fun `gradient() returns the correct gradient`() {
        assertEquals(4.0, OrdinaryLeastSquares.gradient(3.0, 5.0))
    }
}
