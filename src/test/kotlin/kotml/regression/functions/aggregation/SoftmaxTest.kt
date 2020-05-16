package kotml.regression.functions.aggregation

import kotml.math.Vector
import kotml.regression.Weights
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class SoftmaxTest {
    @Test
    fun `aggregate() returns the softmax value of the regressors`() {
        assertEquals(0.244728471054797652, Softmax(1).aggregate(
            Weights(Vector.zeros(3)),
            Vector(5, 6, 7)
        ))
    }

    @Test
    fun `weightsGradient() returns zeros`() {
        assertEquals(
            Weights(0.0, Vector.zeros(3)),
            Softmax(1).weightsGradient(
                Weights(2.0, Vector(-1, 0, 2)),
                Vector(2, 3, 4)
            )
        )
    }

    @Test
    fun `regressorsGradient() returns the derivative at the right index`() {
        val index = 1
        assertEquals(
            0.1848364465099787,
            Softmax(index).regressorsGradient(
                Weights(Vector(-1, 0, 2)),
                Vector(5, 6, 7)
            )[index]
        )
    }
}
