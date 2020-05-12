package kotml.regression.cost

import kotml.math.Vector
import kotml.regression.cost.loss.SquaredError
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class MeanCostTest {
    @Test
    fun `evaluate() returns the correct cost`() {
        assertEquals(12.5, MeanCost(SquaredError).evaluate(
            Vector(1, 5), Vector(-2, 1)))
    }

    @Test
    fun `gradient() returns the correct gradient`() {
        assertEquals(Vector(3, 4), MeanCost(SquaredError).gradient(
            Vector(1, 5), Vector(-2, 1)))
    }
}
