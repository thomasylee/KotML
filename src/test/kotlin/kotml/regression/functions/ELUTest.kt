package kotml.regression.functions

import kotml.math.Vector
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
    fun `evaluate() returns the correct vector`() {
        val elu = ELU(alpha = 0.5)
        assertEquals(
            Vector(-0.43233235838169365, 0.0, 2.0),
            elu.evaluate(Vector(-2, 0, 2)))
    }

    @Test
    fun `derivative() returns the correct value`() {
        val elu = ELU(alpha = 0.5)
        assertEquals(0.06766764161830635, elu.derivative(-2.0))
        assertEquals(1.0, elu.derivative(0.0))
        assertEquals(1.0, elu.derivative(2.0))
    }

    @Test
    fun `derivative() returns the correct vector`() {
        val elu = ELU(alpha = 0.5)
        assertEquals(
            Vector(0.06766764161830635, 1.0, 1.0),
            elu.derivative(Vector(-2, 0, 2)))
    }
}
