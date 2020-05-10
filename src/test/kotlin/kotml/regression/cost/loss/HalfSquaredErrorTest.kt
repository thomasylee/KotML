package kotml.regression.cost.loss

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class HalfSquaredErrorTest {
    @Test
    fun `evaluate() returns the correct cost`() {
        assertEquals(4.5, HalfSquaredError.evaluate(3.0, 6.0))
    }

    @Test
    fun `gradient() returns the correct gradient`() {
        assertEquals(-3.0, HalfSquaredError.gradient(3.0, 6.0))
    }
}
