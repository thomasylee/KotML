package kotml.regression.optimization

import kotml.distributions.DistributionSampler
import kotml.distributions.UniformSampler
import kotml.extensions.* // ktlint-disable no-wildcard-imports
import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.cost.loss.LossFunction
import kotml.regression.functions.FunctionModel
import kotml.regression.functions.aggregation.AggregationFunction
import kotml.regression.functions.aggregation.DotProduct

/**
 * StochasticGradientDescent incrementally estimates weights based on
 * observations as they are provided.
 */
class StochasticGradientDescent(
    val stepSize: Double,
    val function: FunctionModel,
    val lossFunction: LossFunction,
    val weights: Weights,
    val aggregationFunction: AggregationFunction = DotProduct,
    val weightDecayRate: Double = 0.0,
    val weightDecayScalingFactor: Double = 1.0
) : IterativeOptimizer<Weights, Double>(weights.coeffs.shape[0], 1, weights) {
    constructor(
        stepSize: Double,
        function: FunctionModel,
        lossFunction: LossFunction,
        regressorCount: Int,
        hasConstant: Boolean = true,
        aggregationFunction: AggregationFunction = DotProduct,
        sampler: DistributionSampler = UniformSampler(0.0),
        weightDecayRate: Double = 0.0,
        weightDecayScalingFactor: Double = 1.0
    ) : this(
        stepSize = stepSize,
        function = function,
        lossFunction = lossFunction,
        weights = Weights(regressorCount, hasConstant, sampler),
        aggregationFunction = aggregationFunction,
        weightDecayRate = weightDecayRate,
        weightDecayScalingFactor = weightDecayScalingFactor
    )

    protected override fun addObservation(regressors: Vector, targets: Vector) {
        observeAndEvaluate(regressors, targets)
    }

    /**
     * Adjusts weights in batches, where each row of regressorsBatch and
     * targetsBatch is an observation of regressors and targets. The
     * weights are updated as if they were updated at the completion of the
     * batch rather than at the completion of each observation.
     * @param regressorsBatch batch of regressors, shape = (batchSize, numRegressors)
     * @param targetsBatch batch of targets, shape = (batchSize, numTargets)
     */
    override fun batchObserveAndEvaluate(regressorsBatch: Vector, targetsBatch: Vector): Vector {
        val currentWeights = weights.copy()
        return Vector(regressorsBatch.shape[0]) { batchIndex ->
            observeAndEvaluate(
                currentWeights,
                regressorsBatch(batchIndex),
                targetsBatch(batchIndex)
            )
        }
    }

    override fun observeAndEvaluate(evaluatingModel: Weights, regressors: Vector, targets: Vector): Double {
        val estimate = function.evaluate(
            aggregationFunction.aggregate(evaluatingModel, regressors)
        )
        val dF_dIn = lossFunction.derivative(estimate, targets[0])
        val dIn_dWeight = aggregationFunction.weightsGradient(weights, regressors)

        if (weights.hasConstant)
            weights.constant -= stepSize * dF_dIn * dIn_dWeight.constant +
                weightDecayScalingFactor * weightDecayRate * weights.constant

        (0 until regressors.shape[0]).forEach { index ->
            weights.coeffs[index] -= stepSize * dF_dIn * dIn_dWeight.coeffs[index] +
                weightDecayScalingFactor * weightDecayRate * weights.coeffs[index]
        }

        return estimate
    }
}
