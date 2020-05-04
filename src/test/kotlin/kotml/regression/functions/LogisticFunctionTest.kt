package kotml.regression.functions

import kotml.math.Vector
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class LogisticFunctionTest {
    @Test
    fun `estimate() returns the correct estimate value`() {
        assertEquals(0.9046505351008906, LogisticFunction.estimate(
            doubleArrayOf(-4.5, 2.0, 1.5),
            Vector(3.0, 0.5)
        ))
    }

    @Test
    fun `gradient() returns the correct gradient`() {
        assertEquals(
            Vector(0.22534771461105382, 0.3380215719165808),
            LogisticFunction.gradient(
                doubleArrayOf(0.1, -0.5),
                Vector(1.5)
            )
        )
    }
}
