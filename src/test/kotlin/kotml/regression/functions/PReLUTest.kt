package kotml.regression.functions

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class PReLUTest {
    @Test
    fun `evaluate() returns the correct value`() {
        assertEquals(-0.5, PReLU(0.1).evaluate(-5.0))
        assertEquals(0.0, PReLU(0.1).evaluate(0.0))
        assertEquals(5.0, PReLU(0.1).evaluate(5.0))
    }

    @Test
    fun `derivative() returns the correct value`() {
        assertEquals(0.1, PReLU(0.1).derivative(-5.0))
        assertEquals(1.0, PReLU(0.1).derivative(0.0))
        assertEquals(1.0, PReLU(0.1).derivative(5.0))
    }
}
