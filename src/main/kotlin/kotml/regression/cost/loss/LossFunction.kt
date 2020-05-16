package kotml.regression.cost.loss

/**
 * `LossFunction` calculates the loss for a specific data point.
 * A `CostFunction` often (but not always) operates on an aggregation of
 * losses over several data points.
 */
interface LossFunction {
    /**
     * Returns the evaluated value of the loss function.
     * @param estimate estimated value
     * @param target dependent variable value
     * @return loss of the estimate
     */
    abstract fun evaluate(estimate: Double, target: Double): Double

    /**
     * Returns the derivative of the loss function.
     * @param estimate estimated value
     * @param target dependent variable value
     * @return derivative of the loss function
     */
    abstract fun derivative(estimate: Double, target: Double): Double
}
