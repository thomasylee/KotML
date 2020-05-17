package kotml.reinforcement.policies.tabular

import kotml.math.Vector

interface TabularPolicy {
    abstract fun chooseAction(qValuesForState: Vector): Int

    abstract fun actionProbabilities(qValuesForState: Vector): Vector
}
