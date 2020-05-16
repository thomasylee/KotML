package kotml.regression.functions

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class TanhTest {
    @Test
    fun `evaluate() returns the correct value`() {
        assertEquals(0.46211715726000974, Tanh.evaluate(0.5))
        assertEquals(-0.7615941559557649, Tanh.evaluate(-1.0))
    }

    @Test
    fun `derivative() returns the correct value`() {
        assertEquals(0.5965858082813313, Tanh.derivative(0.75))
    }
}
