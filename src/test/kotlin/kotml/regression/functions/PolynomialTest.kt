package kotml.regression.functions

import kotml.math.Vector
import kotml.regression.Weights
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class PolynomialTest {
    @Test
    fun `estimate() returns the correct estimate value with bias`() {
        assertEquals(7.75, Polynomial(1, 2).evaluate(
            Weights(0.5, Vector(-2.5, 1.0)),
            Vector(2.0, 3.5)
        ))
    }

    @Test
    fun `estimate() returns the correct estimate value without bias`() {
        assertEquals(7.25, Polynomial(1, 2).evaluate(
            Weights(Vector(-2.5, 1.0)),
            Vector(2.0, 3.5)
        ))
    }

    @Test
    fun `weightsGradient() returns the correct gradient with bias`() {
        assertEquals(
            Weights(1.0, Vector(2.0, 12.25)),
            Polynomial(1.0, 2.0).weightsGradient(
                Weights(0.5, Vector(-2.5, 1.0)),
                Vector(2.0, 3.5)
            )
        )
    }

    @Test
    fun `weightsGradient() returns the correct gradient without bias`() {
        assertEquals(
            Weights(Vector(2.0, 12.25)),
            Polynomial(1.0, 2.0).weightsGradient(
                Weights(Vector(-2.5, 1.0)),
                Vector(2.0, 3.5)
            )
        )
    }
}
