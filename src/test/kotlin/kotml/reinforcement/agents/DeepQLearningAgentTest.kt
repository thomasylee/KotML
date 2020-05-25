package kotml.reinforcement.agents

import kotlin.random.Random
import kotml.math.Vector
import kotml.regression.functions.IdentityFunction
import kotml.regression.functions.ReLU
import kotml.reinforcement.policies.discrete.EpsilonGreedyPolicy
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class DeepQLearningAgentTest {
    /**
     * Build a 10-step corridor starting at the left end and receiving a
     * reward of 1 at the right end.
     * Actions are (left, right), in order.
     * The policy should learn to move right in every state.
     */
    @Test
    fun `finds optimal solution for 10-step corridor`() {
        val random = Random(0)
        val agent = DeepQLearningAgent(
            stateDimensions = 10,
            discount = 0.9,
            stepSize = 0.0005,
            layerSizes = intArrayOf(10, 2),
            behaviorPolicy = EpsilonGreedyPolicy(random = random),
            targetNetworkUpdateFrequency = 1000,
            random = random
        )
        val numTrials = 20
        assertTrue(
            agent.dqn.network.layers.first().neurons.first().activationFunction is ReLU,
            "First DQN layer should use ReLU"
        )
        assertTrue(
            agent.dqn.network.layers.last().neurons.first().activationFunction is IdentityFunction,
            "Second DQN layer should use IdentityFunction"
        )

        val terminalLoc = 9
        val terminalReward = 1.0
        (1..numTrials).forEach {
            var agentLoc = 0
            var action = agent.start(Vector(10) { if (it == agentLoc) 1.0 else 0.0 })
            var movesLeft = 500
            while (agentLoc != terminalLoc && movesLeft > 0) {
                movesLeft--

                when (action) {
                    0 -> agentLoc = listOf(0, agentLoc - 1).max() ?: -1
                    1 -> agentLoc = agentLoc + 1
                }
                if (agentLoc == terminalLoc) {
                    agent.processTerminalStep(terminalReward)
                } else {
                    action = agent.processStep(0.0, Vector(10) {
                        if (it == agentLoc)
                            1.0
                        else
                            0.0
                    })
                }
            }
            if (movesLeft == 0)
                throw RuntimeException("$it More moves were used than were necessary")
        }

        (0..terminalLoc).forEach { stateIndex ->
            val state = Vector(10) { if (it == stateIndex) 1.0 else 0.0 }
            assertEquals(1, agent.dqn.evaluate(state).argmax(random))
        }
    }
}
