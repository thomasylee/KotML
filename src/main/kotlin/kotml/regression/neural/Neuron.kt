package kotml.regression.neural

import kotml.distributions.DistributionSampler
import kotml.distributions.NormalSampler
import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.functions.FunctionOfLinearRegressors
import kotml.regression.objectives.CostFunction
import kotml.regression.objectives.ObjectiveFunction

/**
 * `Neuron` represents a single neuron in a neural network.
 */
class Neuron(
    val stepSize: Double,
    val activationFunction: FunctionOfLinearRegressors,
    val objectiveFunction: ObjectiveFunction,
    val weights: Weights
) {
    /**
     * Determines whether optimization moves against the gradient (cost
     * function) or with the gradient (reward function).
     */
    private val objectiveFactor =
        if (objectiveFunction is CostFunction)
            -1.0
        else
            1.0

    constructor(
        stepSize: Double,
        activationFunction: FunctionOfLinearRegressors,
        objectiveFunction: ObjectiveFunction,
        regressorCount: Int,
        includeBias: Boolean = true,
        sampler: DistributionSampler = NormalSampler()
    ) : this(
        stepSize = stepSize,
        activationFunction = activationFunction,
        objectiveFunction = objectiveFunction,
        weights = Weights(regressorCount, includeBias, sampler)
    )

    /**
     * Evaluates a vector of inputs and returns the result.
     * @param regressors input values
     * @return output value
     */
    fun evaluate(regressors: Vector): Double =
        activationFunction.evaluate(weights, regressors)
}
