package kotml.regression.functions

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class IdentityFunctionTest {
    @Test
    fun `evaluate() returns the input`() {
        assertEquals(-5.0, IdentityFunction.evaluate(-5.0))
    }

    @Test
    fun `derivative() returns 1`() {
        assertEquals(1.0, IdentityFunction.derivative(-5.0))
    }
}
