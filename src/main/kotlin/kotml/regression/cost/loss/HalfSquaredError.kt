package kotml.regression.cost.loss

/**
 * `HalfSquaredError` calculates loss as half the squared difference between
 * an estimate and the known dependent variable value. Half of the squared
 * error is often used in regression since the derivative has a coefficient
 * of 1.
 */
object HalfSquaredError : LossFunction {
    /**
     * Returns half the squared error.
     * @param estimate estimated value
     * @param target dependent variable value
     * @return half squared error
     */
    override fun evaluate(estimate: Double, target: Double): Double =
        0.5 * (estimate - target) * (estimate - target)

    /**
     * Returns the gradient of half the squared error.
     * @param estimate estimated value
     * @param target dependent variable value
     * @return gradient of half the squared error
     */
    override fun gradient(estimate: Double, target: Double): Double =
        estimate - target
}
