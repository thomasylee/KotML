package kotml.regression

/**
 * RegressionException is thrown when a logical error prevents a regression
 * operation from succeeding.
 */
class RegressionException(override val message: String) : Exception(message)
