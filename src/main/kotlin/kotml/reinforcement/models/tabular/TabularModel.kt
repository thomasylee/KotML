package kotml.reinforcement.models.tabular

import kotml.math.MutableVector

abstract class TabularModel(val numIterations: Int) {
    abstract fun observe(qValues: MutableVector, prevState: Int, prevAction: Int, reward: Double, state: Int)

    protected abstract fun iterate(qValues: MutableVector): Boolean

    fun runIterations(qValues: MutableVector) {
        (1..numIterations).forEach {
            if (!iterate(qValues))
                return
        }
    }
}
