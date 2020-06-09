package kotml.reinforcement.agents

import kotlin.random.Random
import kotml.math.Vector
import kotml.regression.functions.IdentityFunction
import kotml.regression.functions.ReLU
import kotml.regression.neural.DenseNeuralLayer
import kotml.regression.neural.SplitNeuralLayer
import kotml.reinforcement.functionapproximation.dqn.DuelingAggregationFunction
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
    fun trainOn10StepCorridor(agent: DeepQLearningAgent, numTrials: Int) {
        assertTrue(
            (agent.dqn.network.layers.first() as DenseNeuralLayer)
                .neurons.first().activationFunction is ReLU,
            "First DQN layer should use ReLU"
        )
        assertTrue(
            (agent.dqn.network.layers.last() as DenseNeuralLayer)
                .neurons.first().activationFunction is IdentityFunction,
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
                throw RuntimeException("More moves were used than were necessary")
        }
    }

    @Test
    fun `finds optimal solution for 10-step corridor with default network`() {
        val random = Random(0)
        val agent = DeepQLearningAgent(
            stateDimensions = 10,
            discount = 0.9,
            stepSize = 0.001,
            layerSizes = intArrayOf(10, 2),
            behaviorPolicy = EpsilonGreedyPolicy(random = random),
            targetNetworkUpdateFrequency = 1000,
            random = random
        )

        trainOn10StepCorridor(agent, 20)

        (0..9).forEach { stateIndex ->
            val state = Vector(10) { if (it == stateIndex) 1.0 else 0.0 }
            assertEquals(1, agent.dqn.evaluate(state).argmax(random))
        }
    }

    @Test
    fun `finds optimal solution for 10-step corridor with dueling network`() {
        val random = Random(0)
        val agent = DeepQLearningAgent(
            stateDimensions = 10,
            discount = 0.9,
            stepSize = 0.0005,
            layerSizes = intArrayOf(10, 2),
            behaviorPolicy = EpsilonGreedyPolicy(random = random),
            targetNetworkUpdateFrequency = 1000,
            duelingNetwork = true,
            random = random
        )

        // Make sure it's a dueling network.
        assertEquals(
            SplitNeuralLayer::class,
            agent.dqn.network.layers[1]::class
        )
        assertEquals(
            DenseNeuralLayer::class,
            agent.dqn.network.layers.last()::class
        )
        assertEquals(
            DuelingAggregationFunction::class,
            (agent.dqn.network.layers.last() as DenseNeuralLayer).neurons.last().aggregationFunction::class
        )

        trainOn10StepCorridor(agent, 20)

        (0..9).forEach { stateIndex ->
            val state = Vector(10) { if (it == stateIndex) 1.0 else 0.0 }
            assertEquals(1, agent.dqn.evaluate(state).argmax(random))
        }
    }
}
