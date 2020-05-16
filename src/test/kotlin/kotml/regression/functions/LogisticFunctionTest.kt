package kotml.regression.functions

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class LogisticFunctionTest {
    @Test
    fun `evaluate() returns the correct value`() {
        assertEquals(0.9933071490757153, LogisticFunction.evaluate(5.0))
    }

    @Test
    fun `derivative() returns the correct value`() {
        assertEquals(0.19661193324148188, LogisticFunction.derivative(-1.0))
        assertEquals(0.2350037122015945, LogisticFunction.derivative(0.5))
    }
}
