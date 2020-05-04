package kotml.regression.functions

import kotml.math.Vector
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class PolynomialTest {
    @Test
    fun `estimate() returns the correct estimate value with bias`() {
        assertEquals(7.75, Polynomial(1, 2).evaluate(
            doubleArrayOf(0.5, -2.5, 1.0),
            Vector(2.0, 3.5)
        ))
    }

    @Test
    fun `estimate() returns the correct estimate value without bias`() {
        assertEquals(7.25, Polynomial(1, 2).evaluate(
            doubleArrayOf(-2.5, 1.0),
            Vector(2.0, 3.5)
        ))
    }

    @Test
    fun `gradient() returns the correct gradient with bias`() {
        assertEquals(
            Vector(1.0, 2.0, 12.25),
            Polynomial(1.0, 2.0).gradient(
                doubleArrayOf(0.5, -2.5, 1.0),
                Vector(2.0, 3.5)
            )
        )
    }

    @Test
    fun `gradient() returns the correct gradient without bias`() {
        assertEquals(
            Vector(2.0, 12.25),
            Polynomial(1.0, 2.0).gradient(
                doubleArrayOf(-2.5, 1.0),
                Vector(2.0, 3.5)
            )
        )
    }
}
