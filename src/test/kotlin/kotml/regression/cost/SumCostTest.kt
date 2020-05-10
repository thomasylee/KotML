package kotml.regression.cost

import kotml.math.Vector
import kotml.regression.cost.loss.SquaredError
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class SumCostTest {
    @Test
    fun `evaluate() returns the correct cost`() {
        assertEquals(25.0, SumCost(SquaredError).evaluate(
            Vector(1, 5), Vector(-2, 1)))
    }

    @Test
    fun `gradient() returns the correct gradient`() {
        assertEquals(14.0, SumCost(SquaredError).gradient(
            Vector(1, 5), Vector(-2, 1)))
    }
}
