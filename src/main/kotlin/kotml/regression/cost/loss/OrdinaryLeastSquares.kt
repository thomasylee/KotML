package kotml.regression.cost.loss

/**
 * `OrdinaryLeastSquares` calculates loss as the squared difference between
 * an estimate and the known dependent variable value.
 */
object OrdinaryLeastSquares : LossFunction {
    /**
     * Returns the least squares error.
     * @param estimate estimated value
     * @param response dependent variable value
     * @return least squares error
     */
    override fun evaluate(estimate: Double, response: Double): Double =
        (estimate - response) * (estimate - response)

    /**
     * Returns the gradient of the least squares error.
     * @param estimate estimated value
     * @param response dependent variable value
     * @return gradient of the least squares error
     */
    override fun gradient(estimate: Double, response: Double): Double =
        2.0 * (estimate - response)
}
