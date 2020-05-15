package kotml.reinforcement.agents.tabular

import kotlin.random.Random
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class TabularExpectedSarsaAgentTest : TabularAgentTest() {
    @Test
    fun `finds optimal path in 3x3 grid`() {
        val agent = TabularExpectedSarsaAgent(
            numStates = 9,
            numActions = 4,
            epsilon = 0.1,
            stepSize = 0.1,
            discount = 0.9,
            random = Random(0)
        )
        trainOn3By3Grid(agent)

        assertEquals(3, agent.argmax(agent.q(0)))
        assertEquals(3, agent.argmax(agent.q(1)))
        assertEquals(1, agent.argmax(agent.q(2)))
        assertEquals(1, agent.argmax(agent.q(5)))
    }
}
