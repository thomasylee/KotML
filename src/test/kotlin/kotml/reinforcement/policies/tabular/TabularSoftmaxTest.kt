package kotml.reinforcement.policies.tabular

import kotlin.random.Random
import kotml.math.MutableVector
import kotml.math.Vector
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class TabularSoftmaxTest {
    @Test
    fun `chooseAction() selects actions correctly`() {
        val policy = TabularSoftmax(tau = 0.5, random = Random(0))
        val q = Vector(1, 2, 0, 0)
        var actionCounts = MutableVector(0, 0, 0, 0)

        (1..100).forEach { actionCounts[policy.chooseAction(q)]++ }

        assertEquals(Vector(12, 83, 4, 1), actionCounts)
    }

    @Test
    fun `actionProbabilities() returns the correct probabilities`() {
        val policy = TabularSoftmax(tau = 0.5, random = Random(0))

        assertEquals(
            Vector(
                0.11547708589868771,
                0.853266665846437,
                0.015628124127437554,
                0.015628124127437554
            ),
            policy.actionProbabilities(Vector(1, 2, 0, 0))
        )
    }
}
