package kotml.regression.functions

import kotml.math.Vector
import kotml.regression.Weights
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class ReLUTest {
    @Test
    fun `evaluate() returns the correct estimate with bias`() {
        assertEquals(0.0, ReLU.evaluate(
            Weights(-5.0, doubleArrayOf(2.0, -1.0)),
            Vector(3, 2)))
        assertEquals(5.0, ReLU.evaluate(
            Weights(1.0, doubleArrayOf(2.0, -1.0)),
            Vector(3, 2)))
    }

    @Test
    fun `evaluate() returns the correct estimate without bias`() {
        assertEquals(0.0, ReLU.evaluate(
            Weights(false, doubleArrayOf(-2.0, 1.0)),
            Vector(3, 2)))
        assertEquals(4.0, ReLU.evaluate(
            Weights(false, doubleArrayOf(2.0, -1.0)),
            Vector(3, 2)))
    }

    @Test
    fun `gradient() returns the correct gradient with bias`() {
        assertEquals(Weights(0.0, doubleArrayOf(0.0, 0.0)), ReLU.gradient(
            Weights(-5.0, doubleArrayOf(2.0, -1.0)),
            Vector(3, 2)))
        assertEquals(Weights(1.0, doubleArrayOf(3.0, 2.0)), ReLU.gradient(
            Weights(1.0, doubleArrayOf(2.0, -1.0)),
            Vector(3, 2)))
    }

    @Test
    fun `gradient() returns the correct gradient without bias`() {
        assertEquals(Weights(false, doubleArrayOf(0.0, 0.0)), ReLU.gradient(
            Weights(false, doubleArrayOf(-2.0, 1.0)),
            Vector(3, 2)))
        assertEquals(Weights(false, doubleArrayOf(3.0, 2.0)), ReLU.gradient(
            Weights(false, doubleArrayOf(2.0, -1.0)),
            Vector(3, 2)))
    }
}
