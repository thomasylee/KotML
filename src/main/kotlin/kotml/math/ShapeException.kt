package kotml.math

/**
 * ShapeException is thrown when a mathematical operation is attempted
 * on a vector or matrix with an incompatible shape (number of rows
 * and/or columns does not satisfy an operation's constraints).
 */
class ShapeException(override val message: String) : Exception(message)
