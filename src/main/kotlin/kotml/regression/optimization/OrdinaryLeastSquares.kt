package kotml.regression.optimization

import kotml.math.Vector
import kotml.regression.Weights

/**
 * `OrdinaryLeastSquares` computes an estimated model of a linear function
 * by minimizing the residual sum of squares.
 */
class OrdinaryLeastSquares(
    regressorCount: Int,
    val weights: Weights = Weights(regressorCount)
) : BatchOptimizer<Weights>(regressorCount, 1, weights) {
    private val regressorsList = mutableListOf<Vector>()
    private val targetList = mutableListOf<Double>()
    private lateinit var designMatrix: Vector
    private lateinit var targets: Vector
    private var newObservations = true

    constructor(designMatrix: Vector, targets: Vector) : this(designMatrix.shape[1]) {
        this.designMatrix = designMatrix
        this.targets = targets
        (0 until designMatrix.shape[0]).forEach { index ->
            observe(designMatrix(index), Vector(targets[index]))
        }
        newObservations = false
    }

    protected override fun addObservation(regressors: Vector, targets: Vector) {
        newObservations = true
        regressorsList.add(regressors)
        targetList.add(targets[0])
    }

    override fun processBatch() {
        if (!newObservations) return

        newObservations = false
        designMatrix = Vector(regressorsList.size, regressorCount) {
            val sample = it / regressorCount
            regressorsList[sample][it - sample * regressorCount]
        }
        targets = Vector(targetList.size, 1) { targetList[it] }

        val transpose = designMatrix.transpose()
        val params = (transpose x designMatrix).inverse() x transpose x targets
        (0 until weights.coeffs.shape[0]).forEach { index ->
            weights.coeffs[index] = params[index, 0]
        }
    }
}
