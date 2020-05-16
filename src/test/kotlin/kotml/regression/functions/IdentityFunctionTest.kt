package kotml.regression.functions

import kotml.math.Vector
import kotml.regression.Weights
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class IdentityFunctionTest {
    @Test
    fun `evaluateNetInput() returns the net input`() {
        assertEquals(-5.0, IdentityFunction.evaluateNetInput(-5.0))
    }

    @Test
    fun `evaluate() returns the correct estimate with bias`() {
        assertEquals(-1.0, IdentityFunction.evaluate(
            Weights(-5.0, Vector(2, -1)),
            Vector(3, 2)))
    }

    @Test
    fun `evaluate() returns the correct estimate without bias`() {
        assertEquals(4.0, IdentityFunction.evaluate(
            Weights(Vector(2, -1)),
            Vector(3, 2)))
    }

    @Test
    fun `weightsGradient() returns the correct gradient with bias`() {
        assertEquals(Weights(1.0, Vector(3, 2)), IdentityFunction.weightsGradient(
            Weights(-5.0, Vector(2, -1)),
            Vector(3, 2)))
    }

    @Test
    fun `weightsGradient() returns the correct gradient without bias`() {
        assertEquals(Weights(Vector(3, 2)), IdentityFunction.weightsGradient(
            Weights(Vector(-2, 1)),
            Vector(3, 2)))
    }

    @Test
    fun `netInputGradient() returns the correct gradient`() {
        assertEquals(1.0, IdentityFunction.netInputGradient(0.0))
    }
}
