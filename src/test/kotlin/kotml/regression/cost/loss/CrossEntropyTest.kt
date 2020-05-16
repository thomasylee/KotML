package kotml.regression.cost.loss

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class CrossEntropyTest {
    @Test
    fun `evaluate() returns the correct cost`() {
        assertEquals(-5.0, CrossEntropy.evaluate(2.0, 5.0))
    }

    @Test
    fun `derivative() returns the correct value`() {
        assertEquals(-3.6067376022224087, CrossEntropy.derivative(2.0, 5.0))
    }
}
