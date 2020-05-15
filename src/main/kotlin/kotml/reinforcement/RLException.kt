package kotml.reinforcement

/**
 * `RLException` is thrown when a logical error prevents a reinforcement
 * learning operation from succeeding.
 */
class RLException(override val message: String) : Exception(message)
