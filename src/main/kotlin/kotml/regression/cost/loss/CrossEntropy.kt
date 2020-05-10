package kotml.regression.cost.loss

import kotlin.math.log2

/**
 * `CrossEntropy` calculates the cross entropy (log loss) of an estimate
 * and the known dependent variable value. Note that this is distinct
 * from the Kullback-Leibler divergence, which is the different between
 * the cross entropy and entropy of a sample.
 */
object CrossEntropy : LossFunction {
    /**
     * Returns the cross entropy error.
     * @param estimate estimated value
     * @param response dependent variable value
     * @return cross entropy error
     */
    override fun evaluate(estimate: Double, response: Double): Double =
        -response * log2(estimate)

    /**
     * Returns the gradient of the cross entropy error.
     * @param estimate estimated value
     * @param response dependent variable value
     * @return gradient of the cross entropy error
     */
    override fun gradient(estimate: Double, response: Double): Double =
        2.0 * (estimate - response)
}
