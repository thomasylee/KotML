package kotml.regression.cost.loss

import kotlin.math.ln
import kotlin.math.log2

/**
 * `CrossEntropy` calculates the cross entropy (log loss) of an estimate
 * and the known dependent variable value. Note that this is distinct
 * from the Kullback-Leibler divergence, which is the different between
 * the cross entropy and entropy of a sample.
 */
object CrossEntropy : LossFunction {
    private val ln2: Double = ln(2.0)

    /**
     * Returns the cross entropy error.
     * @param estimate estimated value
     * @param target dependent variable value
     * @return cross entropy error
     */
    override fun evaluate(estimate: Double, target: Double): Double =
        -target * log2(estimate)

    /**
     * Returns the derivative of the cross entropy error.
     * @param estimate estimated value
     * @param target dependent variable value
     * @return derivative of the cross entropy error
     */
    override fun derivative(estimate: Double, target: Double): Double =
        -target / (estimate * ln2)
}
