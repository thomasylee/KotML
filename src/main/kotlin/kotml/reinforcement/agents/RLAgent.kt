package kotml.reinforcement.agents

abstract class RLAgent<S, A> {
    abstract fun start(initialState: S): A

    abstract fun processStep(reward: Double, state: S): A

    abstract fun processTerminalStep(reward: Double)
}
