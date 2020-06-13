package kotml.regression.functions

import kotml.math.Vector
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class TanhTest {
    @Test
    fun `evaluate() returns the correct value`() {
        assertEquals(0.46211715726000974, Tanh.evaluate(0.5))
        assertEquals(-0.7615941559557649, Tanh.evaluate(-1.0))
    }

    @Test
    fun `evaluate() returns the correct vector`() {
        assertEquals(
            Vector(0.46211715726000974, -0.7615941559557649),
            Tanh.evaluate(Vector(0.5, -1.0)))
    }

    @Test
    fun `derivative() returns the correct value`() {
        assertEquals(0.5965858082813313, Tanh.derivative(0.75))
    }

    @Test
    fun `derivative() returns the correct vector`() {
        assertEquals(
            Vector(1.0, 0.5965858082813313),
            Tanh.derivative(Vector(0.0, 0.75)))
    }
}
