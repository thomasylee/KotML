package kotml.reinforcement.agents.tabular

import kotlin.random.Random
import kotml.reinforcement.models.tabular.DynaQPlus
import kotml.reinforcement.policies.tabular.TabularEpsilonGreedy
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class TabularQLearningAgentTest : TabularAgentBaseTest() {
    @Test
    fun `finds optimal path in 3x3 grid`() {
        val agent = TabularQLearningAgent(
            numStates = 9,
            numActions = 4,
            stepSize = 0.1,
            discount = 0.9,
            behaviorPolicy = TabularEpsilonGreedy(random = Random(0))
        )
        trainOn3By3Grid(agent, 100)

        assertEquals(3, agent.q(0).argmax())
        assertEquals(3, agent.q(1).argmax())
        assertEquals(1, agent.q(2).argmax())
        assertEquals(1, agent.q(5).argmax())
    }

    @Test
    fun `finds optimal path in 3x3 grid with DynaQPlus`() {
        val agent = TabularQLearningAgent(
            numStates = 9,
            numActions = 4,
            stepSize = 0.1,
            discount = 0.9,
            behaviorPolicy = TabularEpsilonGreedy(random = Random(0)),
            model = DynaQPlus(
                numIterations = 5,
                numStates = 9,
                numActions = 4,
                qUpdate = { prevState, prevAction, reward, state, q ->
                    // Terminal state
                    if (state == -1)
                        0.1 * (reward - q[prevState, prevAction])
                    else
                        0.1 * (reward + 0.9 * q(state).max()[0] - q[prevState, prevAction])
                },
                random = Random(0)
            )
        )
        trainOn3By3Grid(agent, 20)

        assertEquals(3, agent.q(0).argmax())
        assertEquals(3, agent.q(1).argmax())
        assertEquals(1, agent.q(2).argmax())
        assertEquals(1, agent.q(5).argmax())
    }
}
