package kotml.regression.optimization

import kotlin.random.Random
import kotml.TestUtils.assertApproxEquals
import kotml.distributions.UniformSampler
import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.cost.loss.HalfSquaredError
import kotml.regression.cost.loss.SquaredError
import kotml.regression.functions.IdentityFunction
import kotml.regression.functions.aggregation.DotProduct
import kotml.regression.functions.aggregation.Polynomial
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNotEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class StochasticGradientDescentTest {
    @Test
    fun `calculates weights correctly with a constant`() {
        val estimator = StochasticGradientDescent(
            0.001, IdentityFunction, SquaredError, 1, true, Polynomial(Vector(1))
        )
        assertTrue(estimator.weights.hasConstant)
        assertEquals(1, estimator.weights.coeffs.shape[0])
        estimator.observe(Vector(-9), Vector(-19))
        estimator.observe(Vector(-8), Vector(-17))
        estimator.observe(Vector(-7), Vector(-15))
        assertEquals(-0.088344472, estimator.weights.constant)
        assertEquals(0.722863304, estimator.weights.coeffs[0])
        assertEquals(3.525972048, estimator.function.evaluate(
            estimator.aggregationFunction.aggregate(
                estimator.weights, Vector(5.0)
            )
        ))

        val prevWeights = Weights(
            estimator.weights.constant,
            estimator.weights.coeffs.copy()
        )
        assertEquals(3.525972048, estimator.observeAndEvaluate(Vector(5), Vector(5)))
        assertNotEquals(prevWeights, estimator.weights)
    }

    @Test
    fun `calculates weights correctly without a constant`() {
        val estimator = StochasticGradientDescent(
            0.02, IdentityFunction, SquaredError, 1, false, Polynomial(Vector(1))
        )
        estimator.observe(Vector(0.0), Vector(0))
        estimator.observe(Vector(1), Vector(2))
        estimator.observe(Vector(2), Vector(4))
        estimator.observe(Vector(3), Vector(6))
        estimator.observe(Vector(4), Vector(8))
        estimator.observe(Vector(5), Vector(10))
        assertFalse(estimator.weights.hasConstant)
        assertEquals(1, estimator.weights.coeffs.shape[0])
        assertEquals(2.0, estimator.weights.coeffs[0])
        assertEquals(12.0, estimator.function.evaluate(
            estimator.aggregationFunction.aggregate(
                estimator.weights, Vector(6.0)
            )
        ))

        val prevWeights = Weights(estimator.weights.coeffs.copy())
        assertEquals(12.0, estimator.observeAndEvaluate(Vector(6), Vector(15)))
        assertNotEquals(prevWeights, estimator.weights)
    }

    @Test
    fun `providing initWeights sets the weights correctly`() {
        val estimator = StochasticGradientDescent(
            stepSize = 0.001,
            function = IdentityFunction,
            lossFunction = SquaredError,
            weights = Weights(-0.038, Vector(0.342)),
            aggregationFunction = Polynomial(Vector(1))
        )
        estimator.observe(Vector(-8), Vector(-17))
        estimator.observe(Vector(-7), Vector(-15))
        assertEquals(-0.088344472, estimator.weights.constant)
        assertEquals(0.722863304, estimator.weights.coeffs[0])
        assertEquals(3.525972048, estimator.function.evaluate(
            estimator.aggregationFunction.aggregate(
                estimator.weights, Vector(5.0)
            )
        ))
    }

    @Test
    fun `batchObserveAndEvaluate() updates weights after batch`() {
        val estimator = StochasticGradientDescent(
            0.1, IdentityFunction, HalfSquaredError, 1, false, DotProduct, UniformSampler(1.0)
        )
        assertEquals(Vector(1, 2, 3), estimator.batchObserveAndEvaluate(
            Vector(Vector(1), Vector(2), Vector(3)),
            Vector(Vector(2), Vector(4), Vector(6))
        ))
        assertApproxEquals(Vector(2.4, 4.8, 7.2), estimator.batchObserveAndEvaluate(
            Vector(Vector(1), Vector(2), Vector(3)),
            Vector(Vector(2), Vector(4), Vector(6))
        ))
    }

    @Test
    fun `using weight decay converges`() {
        val random = Random(0)
        val estimator = StochasticGradientDescent(
            stepSize = 0.02,
            function = IdentityFunction,
            lossFunction = SquaredError,
            regressorCount = 1,
            hasConstant = true,
            weightDecayRate = 0.0001,
            weightDecayScalingFactor = 1.0
        )
        (0..200).shuffled(random).forEach { intX ->
            val x = intX.toDouble() / 200.0
            estimator.observe(Vector(x), Vector(2 * x + 1))
        }
        assertApproxEquals(
            2.0,
            estimator.function.evaluate(
                estimator.aggregationFunction.aggregate(
                    estimator.weights, Vector(0.5)
                )
            ),
            0.01
        )
    }
}
