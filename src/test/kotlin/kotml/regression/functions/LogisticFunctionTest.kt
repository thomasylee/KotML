package kotml.regression.functions

import kotml.math.Vector
import kotml.regression.Weights
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class LogisticFunctionTest {
    @Test
    fun `estimate() returns the correct estimate value with bias`() {
        assertEquals(0.9046505351008906, LogisticFunction.evaluate(
            Weights(-4.5, doubleArrayOf(2.0, 1.5)),
            Vector(3.0, 0.5)
        ))
    }

    @Test
    fun `estimate() returns the correct estimate value without bias`() {
        assertEquals(0.9988304897349445, LogisticFunction.evaluate(
            Weights(false, doubleArrayOf(2.0, 1.5)),
            Vector(3.0, 0.5)
        ))
    }

    @Test
    fun `gradient() returns the correct gradient with bias`() {
        assertEquals(
            Weights(0.22534771461105382, doubleArrayOf(0.3380215719165808)),
            LogisticFunction.gradient(
                Weights(0.1, doubleArrayOf(-0.5)),
                Vector(1.5)
            )
        )
    }

    @Test
    fun `gradient() returns the correct gradient without bias`() {
        assertEquals(
            Weights(false, doubleArrayOf(0.32684249064272103)),
            LogisticFunction.gradient(
                Weights(false, doubleArrayOf(-0.5)),
                Vector(1.5)
            )
        )
    }
}
