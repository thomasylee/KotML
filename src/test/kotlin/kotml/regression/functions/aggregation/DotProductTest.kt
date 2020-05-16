package kotml.regression.functions.aggregation

import kotml.math.Vector
import kotml.regression.Weights
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class DotProductTest {
    @Test
    fun `aggregate() returns linear combination of weights and regressors`() {
        assertEquals(11.0, DotProduct.aggregate(
            Weights(Vector(0, 1, 2)),
            Vector(2, 3, 4)
        ))

        assertEquals(12.0, DotProduct.aggregate(
            Weights(1.0, Vector(0, 1, 2)),
            Vector(2, 3, 4)
        ))
    }

    @Test
    fun `weightsGradient() returns the derivative with respect to weights`() {
        assertEquals(
            Weights(1.0, Vector(2, 3, 4)),
            DotProduct.weightsGradient(
                Weights(2.0, Vector(-1, 0, 2)),
                Vector(2, 3, 4)
            )
        )
    }

    @Test
    fun `regressorsGradient() returns the derivative with respect to regressors`() {
        assertEquals(
            Vector(-1, 0, 2),
            DotProduct.regressorsGradient(
                Weights(2.0, Vector(-1, 0, 2)),
                Vector(2, 3, 4)
            )
        )
    }
}
