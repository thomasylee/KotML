package kotml.reinforcement.agents.tabular

import kotlin.random.Random
import kotml.reinforcement.models.tabular.DynaQPlus
import kotml.reinforcement.models.tabular.PrioritizedDynaQPlus
import kotml.reinforcement.policies.discrete.EpsilonGreedyPolicy
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class TabularQLearningAgentTest : TabularAgentBaseTest() {
    @Test
    fun `finds optimal path in 3x3 grid`() {
        val random = Random(0)
        val agent = TabularQLearningAgent(
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
        val agent = TabularQLearningAgent(
            numStates = 9,
            numActions = 4,
            stepSize = 0.1,
            discount = 0.9,
            behaviorPolicy = EpsilonGreedyPolicy(random = random),
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
                random = random
            )
        )
        trainOn3By3Grid(agent, 20)

        assertEquals(3, agent.q(0).argmax(random))
        assertEquals(3, agent.q(1).argmax(random))
        assertEquals(1, agent.q(2).argmax(random))
        assertEquals(1, agent.q(5).argmax(random))
    }

    @Test
    fun `finds optimal path in 10-step corridor with DynaQPlus`() {
        val random = Random(0)
        val agent = TabularQLearningAgent(
            numStates = 10,
            numActions = 2,
            stepSize = 0.1,
            discount = 0.9,
            behaviorPolicy = EpsilonGreedyPolicy(random = random),
            model = DynaQPlus(
                numIterations = 10,
                numStates = 10,
                numActions = 2,
                qUpdate = { prevState, prevAction, reward, state, q ->
                    // Terminal state
                    if (state == -1)
                        0.1 * (reward - q[prevState, prevAction])
                    else
                        0.1 * (reward + 0.9 * q(state).max()[0] - q[prevState, prevAction])
                },
                random = random
            )
        )
        trainOn10StepCorridor(agent, 10)

        (0 until 10).forEach { state ->
            assertEquals(1, agent.q(state).argmax(random))
        }
    }

    @Test
    fun `finds optimal path in 10-step corridor with PrioritizedDynaQPlus`() {
        val random = Random(0)
        val agent = TabularQLearningAgent(
            numStates = 10,
            numActions = 2,
            stepSize = 0.1,
            discount = 0.9,
            behaviorPolicy = EpsilonGreedyPolicy(random = random),
            model = PrioritizedDynaQPlus(
                numIterations = 5,
                numStates = 10,
                numActions = 2,
                qUpdate = { prevState, prevAction, reward, state, q ->
                    // Terminal state
                    if (state == -1)
                        0.1 * (reward - q[prevState, prevAction])
                    else
                        0.1 * (reward + 0.9 * q(state).max()[0] - q[prevState, prevAction])
                },
                random = random
            )
        )
        trainOn10StepCorridor(agent, 5)

        (0 until 10).forEach { state ->
            assertEquals(1, agent.q(state).argmax(random))
        }
    }
}
