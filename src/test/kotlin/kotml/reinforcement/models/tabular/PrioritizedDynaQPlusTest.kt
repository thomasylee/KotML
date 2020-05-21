package kotml.reinforcement.models.tabular

import kotlin.random.Random
import kotml.math.MutableVector
import kotml.math.Vector
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class PrioritizedDynaQPlusTest {
    @Test
    fun `performs DynaQ+ model updates correctly`() {
        val numStates = 3
        val numActions = 2
        val qValues = MutableVector.zeros(numStates, numActions)
        val model = PrioritizedDynaQPlus(
            numIterations = 1,
            numStates = numStates,
            numActions = numActions,
            qUpdate = { prevState, prevAction, reward, state, q ->
                reward + 0.5 * q(state).max()[0] - q[prevState, prevAction]
            },
            random = Random(0)
        )

        model.observe(qValues, 0, 0, 8.0, 1)
        model.runIterations(qValues)

        assertEquals(Vector(Vector(8, 0), Vector(0, 0), Vector(0, 0)), qValues)

        model.observe(qValues, 1, 0, 4.0, 2)
        model.runIterations(qValues)

        assertEquals(Vector(Vector(8, 0), Vector(4, 0), Vector(0, 0)), qValues)

        model.observe(qValues, 2, 1, 2.0, 1)
        model.runIterations(qValues)

        assertEquals(Vector(Vector(10, 0), Vector(4, 0), Vector(0, 0)), qValues)
        model.runIterations(qValues)

        model.observe(qValues, 1, 1, 10.0, 0)
        model.runIterations(qValues)
        assertEquals(Vector(Vector(10, 0), Vector(6, 0), Vector(0, 4)), qValues)
        model.runIterations(qValues)
        assertEquals(Vector(Vector(11, 0), Vector(6, 0), Vector(0, 4)), qValues)
        model.runIterations(qValues)
        assertEquals(Vector(Vector(11, 0), Vector(6, 0), Vector(0, 5)), qValues)
    }
}
