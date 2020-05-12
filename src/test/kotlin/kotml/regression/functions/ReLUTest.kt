package kotml.regression.functions

import kotml.math.Vector
import kotml.regression.Weights
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class ReLUTest {
    @Test
    fun `evaluateNetInput() returns the correct estimate`() {
        assertEquals(0.0, ReLU.evaluateNetInput(-1.0))
        assertEquals(0.0, ReLU.evaluateNetInput(0.0))
        assertEquals(1.0, ReLU.evaluateNetInput(1.0))
    }

    @Test
    fun `evaluate() returns the correct estimate with bias`() {
        assertEquals(0.0, ReLU.evaluate(
            Weights(-5.0, Vector(2.0, -1.0)),
            Vector(3, 2)))
        assertEquals(5.0, ReLU.evaluate(
            Weights(1.0, Vector(2.0, -1.0)),
            Vector(3, 2)))
    }

    @Test
    fun `evaluate() returns the correct estimate without bias`() {
        assertEquals(0.0, ReLU.evaluate(
            Weights(Vector(-2.0, 1.0)),
            Vector(3, 2)))
        assertEquals(4.0, ReLU.evaluate(
            Weights(Vector(2.0, -1.0)),
            Vector(3, 2)))
    }

    @Test
    fun `weightsGradient() returns the correct gradient with bias`() {
        assertEquals(Weights(0.0, Vector(0.0, 0.0)), ReLU.weightsGradient(
            Weights(-5.0, Vector(2.0, -1.0)),
            Vector(3, 2)))
        assertEquals(Weights(1.0, Vector(3.0, 2.0)), ReLU.weightsGradient(
            Weights(1.0, Vector(2.0, -1.0)),
            Vector(3, 2)))
    }

    @Test
    fun `weightsGradient() returns the correct gradient without bias`() {
        assertEquals(Weights(Vector(0.0, 0.0)), ReLU.weightsGradient(
            Weights(Vector(-2.0, 1.0)),
            Vector(3, 2)))
        assertEquals(Weights(Vector(3.0, 2.0)), ReLU.weightsGradient(
            Weights(Vector(2.0, -1.0)),
            Vector(3, 2)))
    }

    @Test
    fun `netInputGradient() returns the correct gradient`() {
        assertEquals(0.0, ReLU.netInputGradient(-5.0))
        assertEquals(1.0, ReLU.netInputGradient(0.0))
        assertEquals(1.0, ReLU.netInputGradient(5.0))
    }
}
