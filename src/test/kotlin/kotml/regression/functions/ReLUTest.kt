package kotml.regression.functions

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class ReLUTest {
    @Test
    fun `evaluate() returns the correct value`() {
        assertEquals(0.0, ReLU.evaluate(-5.0))
        assertEquals(0.0, ReLU.evaluate(0.0))
        assertEquals(5.0, ReLU.evaluate(5.0))
    }

    @Test
    fun `derivative() returns the correct value`() {
        assertEquals(0.0, ReLU.derivative(-5.0))
        assertEquals(1.0, ReLU.derivative(0.0))
        assertEquals(1.0, ReLU.derivative(5.0))
    }
}
