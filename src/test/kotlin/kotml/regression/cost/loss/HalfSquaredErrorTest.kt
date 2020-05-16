package kotml.regression.cost.loss

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class HalfSquaredErrorTest {
    @Test
    fun `evaluate() returns the correct cost`() {
        assertEquals(4.5, HalfSquaredError.evaluate(3.0, 6.0))
    }

    @Test
    fun `derivative() returns the correct value`() {
        assertEquals(-3.0, HalfSquaredError.derivative(3.0, 6.0))
    }
}
