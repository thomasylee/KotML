package kotml.regression.optimization

import kotml.math.ShapeException
import kotml.math.Vector
import kotml.regression.RegressionException
import kotml.regression.functions.FunctionModel
import kotml.regression.objectives.ObjectiveFunction

/**
 * An Optimizer develops a model of any kind of linear function by optimizing
 * an objective function.
 */
abstract class WeightedOptimizer(
    val regressorCount: Int,
    val function: FunctionModel,
    val objectiveFunction: ObjectiveFunction,
    val includeBias: Boolean,
    val weights: DoubleArray = DoubleArray(
        regressorCount + if (includeBias) 1 else 0
    )
) {
    init {
        if (regressorCount < 1) {
            throw ShapeException("regressorCount must be at least 1")
        }
        val weightCount = regressorCount + if (includeBias) 1 else 0
        if (weights.size != weightCount) {
            throw RegressionException(
                "Number of weights ${weights.size} was expected to be $weightCount"
            )
        }
    }

    constructor(
        regressorCount: Int,
        function: FunctionModel,
        objectiveFunction: ObjectiveFunction,
        weights: DoubleArray = DoubleArray(regressorCount + 1)
    ) : this(regressorCount, function, objectiveFunction, weights.size > regressorCount, weights)

    /**
     * Adds an observation to the training model.
     * @param response the dependent variable value
     * @param regressors the independent variables
     */
    fun observe(regressors: Vector, response: Double) {
        validateRegressorsShape(regressors)
        addObservation(response, regressors)
    }

    internal abstract fun addObservation(response: Double, regressors: Vector)

    internal fun validateRegressorsShape(regressors: Vector) {
        if (regressors.dimensions != 1 || regressors.shape[0] != regressorCount) {
            throw RegressionException(
                "Shape of regressors [${regressors.shape.joinToString(", ")}] must be [$regressorCount]"
            )
        }
    }
}
