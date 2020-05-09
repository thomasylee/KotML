package kotml.regression.cost

interface CostFunction {
    /**
     * Returns the evaluated value of the objective function.
     * @param response dependent variable value
     * @param estimate estimated value
     * @return least squares error
     */
    abstract fun evaluate(response: Double, estimate: Double): Double

    /**
     * Returns the gradient of the objective function.
     * @param response dependent variable value
     * @param estimate estimated value
     * @return gradient of the least squares error
     */
    abstract fun gradient(response: Double, estimate: Double): Double
}
