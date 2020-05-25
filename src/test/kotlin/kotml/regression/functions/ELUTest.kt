package kotml.regression.functions

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class ELUTest {
    @Test
    fun `evaluate() returns the correct value`() {
        val elu = ELU(alpha = 0.5)
        assertEquals(-0.43233235838169365, elu.evaluate(-2.0))
        assertEquals(0.0, elu.evaluate(0.0))
        assertEquals(2.0, elu.evaluate(2.0))
    }

    @Test
    fun `derivative() returns the correct value`() {
        val elu = ELU(alpha = 0.5)
        assertEquals(0.06766764161830635, elu.derivative(-2.0))
        assertEquals(1.0, elu.derivative(0.0))
        assertEquals(1.0, elu.derivative(2.0))
    }
}
