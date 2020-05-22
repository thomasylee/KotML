package kotml.reinforcement.policies

import kotlin.random.Random
import kotml.math.Vector

abstract class BehaviorPolicy<A>(val random: Random = Random) {
    /**
     * Returns a chosen action or actions based on the action values.
     * @param qValuesForState action values
     * @return chosen action or actions
     */
    abstract fun chooseAction(qValuesForState: Vector): A
}
