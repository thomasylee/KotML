package kotml.regression.objectives

object OrdinaryLeastSquares : CostFunction {
    /**
     * Returns the least squares error.
     * @param response dependent variable value
     * @param estimate estimated value
     * @return least squares error
     */
    override fun evaluate(response: Double, estimate: Double): Double =
        (estimate - response) * (estimate - response)

    /**
     * Returns the gradient of the least squares cost at a given point in
     * the provided estimating function.
     * @param response dependent variable value
     * @param estimate estimated value
     * @return gradient of the least squares error
     */
    override fun gradient(response: Double, estimate: Double): Double =
        2.0 * (estimate - response)
}
