package kotml.regression.functions

import kotml.math.Vector
import kotml.regression.Weights
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class TanhTest {
    @Test
    fun `evaluateNetInput() returns the correct estimate`() {
        assertEquals(0.46211715726000974, Tanh.evaluateNetInput(0.5))
        assertEquals(-0.7615941559557649, Tanh.evaluateNetInput(-1.0))
    }

    @Test
    fun `evaluate() returns the correct estimate with bias`() {
        assertEquals(-0.7615941559557649, Tanh.evaluate(
            Weights(-1.5, Vector(2.0, -3.0)),
            Vector(0.25, 0.0)))
    }

    @Test
    fun `evaluate() returns the correct estimate without bias`() {
        assertEquals(0.46211715726000974, Tanh.evaluate(
            Weights(Vector(2.0, -3.0)),
            Vector(0.25, 0.0)))
    }

    @Test
    fun `weightsGradient() returns the correct gradient with bias`() {
        assertEquals(
            Weights(0.41997434161402597, Vector(1.2599230248420779)),
            Tanh.weightsGradient(
                Weights(-5.0, Vector(2.0)),
                Vector(3)))
    }

    @Test
    fun `weightsGradient() returns the correct gradient without bias`() {
        assertEquals(Weights(Vector(0.0000737296422159981)), Tanh.weightsGradient(
            Weights(Vector(2.0)),
            Vector(3)))
    }

    @Test
    fun `netInputGradient() returns the correct gradient`() {
        assertEquals(0.5965858082813313, Tanh.netInputGradient(0.75))
    }
}
