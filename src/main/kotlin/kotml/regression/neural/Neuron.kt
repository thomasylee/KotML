package kotml.regression.neural

import kotml.math.Vector
import kotml.regression.optimization.WeightedOptimizer

/**
 * `Neuron` represents a single neuron in a neural network. It uses a
 * weighted optimizer to update weights to optimize an objective function.
 */
class Neuron(
    // Optimizer is mutable so it can be swapped with a different optimizer
    // at any time.
    var optimizer: WeightedOptimizer
) {
    /**
     * Evaluates a vector of inputs and returns the result.
     * @param regressors input values
     * @return output value
     */
    fun evaluate(regressors: Vector): Double =
        optimizer.function.evaluate(optimizer.weights, regressors)
}
