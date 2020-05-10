package kotml.regression.cost.loss

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class CrossEntropyTest {
    @Test
    fun `evaluate() returns the correct cost`() {
        assertEquals(-5.0, CrossEntropy.evaluate(2.0, 5.0))
    }

    @Test
    fun `gradient() returns the correct gradient`() {
        assertEquals(-3.6067376022224087, CrossEntropy.gradient(2.0, 5.0))
    }
}
