package kotml.regression.cost.loss

/**
 * `SquaredError` calculates loss as the squared difference between an
 * estimate and the known dependent variable value, also referred to as
 * the squared Euclidian distance.
 */
object SquaredError : LossFunction {
    /**
     * Returns the squared error.
     * @param estimate estimated value
     * @param target dependent variable value
     * @return squared error
     */
    override fun evaluate(estimate: Double, target: Double): Double =
        (estimate - target) * (estimate - target)

    /**
     * Returns the gradient of the squared error.
     * @param estimate estimated value
     * @param target dependent variable value
     * @return gradient of the squared error
     */
    override fun gradient(estimate: Double, target: Double): Double =
        2.0 * (estimate - target)
}
