package kotml.reinforcement.policies.tabular

import kotlin.random.Random
import kotml.math.MutableVector
import kotml.math.Vector
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class TabularEpsilonGreedyTest {
    @Test
    fun `chooseAction() selects actions correctly`() {
        val policy = TabularEpsilonGreedy(epsilon = 0.2, random = Random(0))
        val q = Vector(0, 1, 0, 0)
        var actionCounts = MutableVector(0, 0, 0, 0)

        (1..100).forEach { actionCounts[policy.chooseAction(q)]++ }

        assertEquals(Vector(2, 82, 7, 9), actionCounts)
    }

    @Test
    fun `actionProbabilities() returns the correct probabilities`() {
        val policy = TabularEpsilonGreedy(epsilon = 0.2, random = Random(0))

        assertEquals(
            Vector(0.05, 0.85, 0.05, 0.05),
            // Account for floating point arithmetic errors.
            policy.actionProbabilities(Vector(0, 1, 0, 0)).map { value ->
                (1000.0 * value).toInt() / 1000.0
            }
        )
    }
}
