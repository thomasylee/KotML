package kotml.reinforcement.agents.tabular

import kotml.math.Vector

abstract class TabularAgentBaseTest {
    /**
     * Build a 3x3 grid with the agent starting in the top left (0, 0).
     * Make the reward less negative along the top row and rightmost column.
     * Actions are (up, down, left, right), in order.
     * The policy should learn to move right twice and down twice to reach
     * the terminal state in the bottom right (2, 2).
     */
    fun trainOn3By3Grid(agent: TabularAgent, numTrials: Int) {
        val rewards = Vector(
            Vector(-1, -1, -1),
            Vector(-4, -5, -2),
            Vector(-1, -3, -1))

        data class Loc(var row: Int, var col: Int) {
            fun toState(): Int = row * 3 + col
        }

        (1..numTrials).forEach {
            var agentLoc = Loc(0, 0)
            var action = agent.start(agentLoc.toState())
            val terminalLoc = Loc(2, 2)
            var movesLeft = 100
            while (agentLoc != terminalLoc && movesLeft > 0) {
                movesLeft--

                when (action) {
                    0 -> agentLoc.row = listOf(agentLoc.row - 1, 0).max() ?: -1
                    1 -> agentLoc.row = listOf(agentLoc.row + 1, 2).min() ?: -1
                    2 -> agentLoc.col = listOf(agentLoc.col - 1, 0).max() ?: -1
                    3 -> agentLoc.col = listOf(agentLoc.col + 1, 2).min() ?: -1
                }
                val reward = rewards[agentLoc.row, agentLoc.col]
                if (agentLoc == terminalLoc) {
                    agent.processTerminalStep(reward)
                } else {
                    action = agent.processStep(reward, (agentLoc).toState())
                }
            }
            if (movesLeft == 0)
                throw RuntimeException("More moves were used than were necessary")
        }
    }
}
