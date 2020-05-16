package kotml.regression.cost.loss

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class SquaredErrorTest {
    @Test
    fun `evaluate() returns the correct cost`() {
        assertEquals(9.0, SquaredError.evaluate(3.0, 6.0))
    }

    @Test
    fun `derivative() returns the correct value`() {
        assertEquals(-6.0, SquaredError.derivative(3.0, 6.0))
    }
}
