package kotml.regression.functions.aggregation

import kotml.math.Vector
import kotml.regression.Weights
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class PolynomialTest {
    @Test
    fun `aggregate() returns the correct aggregation with bias`() {
        assertEquals(7.75, Polynomial(1, 2).aggregate(
            Weights(0.5, Vector(-2.5, 1.0)),
            Vector(2.0, 3.5)
        ))
    }

    @Test
    fun `aggregate() returns the correct aggregation without bias`() {
        assertEquals(7.25, Polynomial(1, 2).aggregate(
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

    @Test
    fun `regressorsGradient() returns the correct gradient`() {
        assertEquals(
            Vector(-0.125, 0.0, -2.0, 42.0),
            Polynomial(-1, 0, 1, 2).regressorsGradient(
                Weights(5.0, Vector(2, 1, -2, 3)),
                Vector(4, 5, 6, 7)
            )
        )
    }
}
