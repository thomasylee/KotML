package kotml.regression.functions

import kotml.math.Vector
import kotml.regression.Weights
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class FunctionOfLinearRegressorsTest {
    private object TestFunction : FunctionOfLinearRegressors {
        override fun evaluateNetInput(netInput: Double): Double = 0.0
        override fun netInputGradient(netInput: Double): Double = 0.0
        override fun weightsGradient(weights: Weights, regressors: Vector): Weights = weights
    }

    @Test
    fun `calculateNetInput() returns linear combination of weights and regressors`() {
        assertEquals(11.0, TestFunction.calculateNetInput(
            Weights(Vector(0, 1, 2)),
            Vector(2, 3, 4)
        ))

        assertEquals(12.0, TestFunction.calculateNetInput(
            Weights(1.0, Vector(0, 1, 2)),
            Vector(2, 3, 4)
        ))
    }
}
