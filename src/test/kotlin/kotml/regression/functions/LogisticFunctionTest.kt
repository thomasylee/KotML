package kotml.regression.functions

import kotml.math.Vector
import kotml.regression.Weights
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class LogisticFunctionTest {
    @Test
    fun `estimateNetInput() returns the correct estimate value`() {
        assertEquals(0.9933071490757153, LogisticFunction.evaluateNetInput(5.0))
    }

    @Test
    fun `estimate() returns the correct estimate value with bias`() {
        assertEquals(0.9046505351008906, LogisticFunction.evaluate(
            Weights(-4.5, Vector(2.0, 1.5)),
            Vector(3.0, 0.5)
        ))
    }

    @Test
    fun `estimate() returns the correct estimate value without bias`() {
        assertEquals(0.9988304897349445, LogisticFunction.evaluate(
            Weights(Vector(2.0, 1.5)),
            Vector(3.0, 0.5)
        ))
    }

    @Test
    fun `weightsGradient() returns the correct gradient with bias`() {
        assertEquals(
            Weights(0.22534771461105382, Vector(0.3380215719165808)),
            LogisticFunction.weightsGradient(
                Weights(0.1, Vector(-0.5)),
                Vector(1.5)
            )
        )
    }

    @Test
    fun `weightsGradient() returns the correct gradient without bias`() {
        assertEquals(
            Weights(Vector(0.32684249064272103)),
            LogisticFunction.weightsGradient(
                Weights(Vector(-0.5)),
                Vector(1.5)
            )
        )
    }

    @Test
    fun `netInputGradient() returns the correct gradient`() {
        assertEquals(0.19661193324148188, LogisticFunction.netInputGradient(-1.0))
        assertEquals(0.2350037122015945, LogisticFunction.netInputGradient(0.5))
    }
}
