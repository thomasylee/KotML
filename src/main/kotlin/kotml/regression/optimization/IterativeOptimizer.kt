package kotml.regression.optimization

import kotml.regression.Weights
import kotml.regression.cost.loss.LossFunction
import kotml.regression.functions.FunctionModel

/**
 * IterativeOptimizer develops a model of any kind of linear function by
 * iteratively reducing a loss function.
 */
abstract class IterativeOptimizer(
    function: FunctionModel,
    val lossFunction: LossFunction,
    weights: Weights
) : Optimizer(function, weights)
