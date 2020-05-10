package kotml.regression.functions

import kotml.math.Vector
import kotml.regression.Weights
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class LinearFunctionTest {
    @Test
    fun `evaluate() returns the correct estimate with bias`() {
        assertEquals(-1.0, LinearFunction.evaluate(
            Weights(-5.0, Vector(2, -1)),
            Vector(3, 2)))
    }

    @Test
    fun `evaluate() returns the correct estimate without bias`() {
        assertEquals(4.0, LinearFunction.evaluate(
            Weights(Vector(2, -1)),
            Vector(3, 2)))
    }

    @Test
    fun `weightsGradient() returns the correct gradient with bias`() {
        assertEquals(Weights(1.0, Vector(3, 2)), LinearFunction.weightsGradient(
            Weights(-5.0, Vector(2, -1)),
            Vector(3, 2)))
    }

    @Test
    fun `weightsGradient() returns the correct gradient without bias`() {
        assertEquals(Weights(Vector(3, 2)), LinearFunction.weightsGradient(
            Weights(Vector(-2, 1)),
            Vector(3, 2)))
    }

    @Test
    fun `netInputGradient() returns the correct gradient`() {
        assertEquals(1.0, LinearFunction.netInputGradient(0.0))
    }
}
