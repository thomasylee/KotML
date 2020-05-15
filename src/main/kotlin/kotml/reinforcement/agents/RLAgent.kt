package kotml.reinforcement.agents

import kotml.math.Vector

abstract class RLAgent<S, A> {
    abstract fun start(initialState: S): A

    abstract fun processStep(reward: Double, state: S): A

    abstract fun processTerminalStep(reward: Double)

    fun argmax(vector: Vector): Int {
        var maxValue = -Double.MAX_VALUE
        val maxIndices = mutableListOf<Int>()
        vector.forEachIndexed { index, value ->
            if (value > maxValue) {
                maxIndices.clear()
                maxIndices.add(index)
                maxValue = value
            } else if (value == maxValue) {
                maxIndices.add(index)
            }
        }
        return maxIndices.random()
    }
}
