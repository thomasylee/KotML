package kotml.reinforcement.agents.tabular

import kotlin.random.Random
import kotml.reinforcement.models.tabular.DynaQPlus
import kotml.reinforcement.policies.discrete.EpsilonGreedyPolicy
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class TabularExpectedSarsaAgentTest : TabularAgentBaseTest() {
    @Test
    fun `finds optimal path in 3x3 grid`() {
        val random = Random(0)
        val agent = TabularExpectedSarsaAgent(
            numStates = 9,
            numActions = 4,
            stepSize = 0.1,
            discount = 0.9,
            behaviorPolicy = EpsilonGreedyPolicy(random = random)
        )
        trainOn3By3Grid(agent, 100)

        assertEquals(3, agent.q(0).argmax(random))
        assertEquals(3, agent.q(1).argmax(random))
        assertEquals(1, agent.q(2).argmax(random))
        assertEquals(1, agent.q(5).argmax(random))
    }

    @Test
    fun `finds optimal path in 3x3 grid with DynaQPlus`() {
        val random = Random(0)
        val agent = TabularExpectedSarsaAgent(
            numStates = 9,
            numActions = 4,
            stepSize = 0.1,
            discount = 0.9,
            behaviorPolicy = EpsilonGreedyPolicy(random = random)
        )
        agent.model = DynaQPlus(
            numIterations = 5,
            numStates = 9,
            numActions = 4,
            qUpdate = { prevState, prevAction, reward, state, q ->
                // Terminal state
                if (state == -1)
                    0.1 * (reward - q[prevState, prevAction])
                else
                    0.1 * (reward + 0.9 * agent.expectedQ(state) - q[prevState, prevAction])
            },
            random = random
        )
        trainOn3By3Grid(agent, 20)

        assertEquals(3, agent.q(0).argmax(random))
        assertEquals(3, agent.q(1).argmax(random))
        assertEquals(1, agent.q(2).argmax(random))
        assertEquals(1, agent.q(5).argmax(random))
    }
}
