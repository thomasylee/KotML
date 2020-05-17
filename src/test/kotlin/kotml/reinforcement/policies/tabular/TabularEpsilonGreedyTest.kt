package kotml.reinforcement.policies.tabular

import kotlin.random.Random
import kotml.math.MutableVector
import kotml.math.Vector
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class TabularEpsilonGreedyTest {
    @Test
    fun `selects actions correctly`() {
        val policy = TabularEpsilonGreedy(epsilon = 0.2, random = Random(0))
        val q = Vector(0, 1, 0, 0)
        var actionCounts = MutableVector(0, 0, 0, 0)

        (1..100).forEach { actionCounts[policy.chooseAction(q)]++ }

        assertEquals(Vector(2, 82, 7, 9), actionCounts)
    }
}
