package kotml.regression.cost.loss

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class SquaredErrorTest {
    @Test
    fun `evaluate() returns the correct cost`() {
        assertEquals(9.0, SquaredError.evaluate(3.0, 6.0))
    }

    @Test
    fun `gradient() returns the correct gradient`() {
        assertEquals(-6.0, SquaredError.gradient(3.0, 6.0))
    }
}
