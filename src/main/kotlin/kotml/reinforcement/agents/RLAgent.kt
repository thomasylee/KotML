package kotml.reinforcement.agents

import kotml.reinforcement.policies.BehaviorPolicy

abstract class RLAgent<S, A>(val behaviorPolicy: BehaviorPolicy<A>) {
    abstract fun start(initialState: S): A

    abstract fun processStep(reward: Double, state: S): A

    abstract fun processTerminalStep(reward: Double)
}
