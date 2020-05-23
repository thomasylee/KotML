package kotml.regression.optimization.backpropagation

import kotlin.math.pow
import kotlin.random.Random
import kotml.TestUtils.assertApproxEquals
import kotml.distributions.NormalSampler
import kotml.distributions.UniformSampler
import kotml.math.Vector
import kotml.regression.cost.MeanCost
import kotml.regression.cost.SumCost
import kotml.regression.cost.loss.HalfSquaredError
import kotml.regression.cost.loss.SquaredError
import kotml.regression.functions.IdentityFunction
import kotml.regression.functions.LogisticFunction
import kotml.regression.functions.Tanh
import kotml.regression.neural.FeedforwardNeuralNetwork
import kotml.regression.neural.NeuralLayer
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class AdamBackpropagationTest {
    @Test
    fun `calculates weights correctly with linear functions`() {
        val network = FeedforwardNeuralNetwork(
            stepSize = 0.01,
            layers = arrayOf(
                NeuralLayer(
                    neuronCount = 1,
                    activationFunction = IdentityFunction,
                    regressorCount = 1,
                    sampler = UniformSampler(0.1)),
                NeuralLayer(
                    neuronCount = 2,
                    activationFunction = IdentityFunction,
                    regressorCount = 1,
                    sampler = UniformSampler(0.1)),
                NeuralLayer(
                    neuronCount = 1,
                    activationFunction = IdentityFunction,
                    regressorCount = 2,
                    sampler = UniformSampler(0.1))
            )
        )

        val optimizer = AdamBackpropagation(
            network = network,
            costFunction = SumCost(SquaredError)
        )

        val rand = Random(0)
        (-700..700).shuffled(rand).forEach { intX ->
            val x = intX.toDouble() / 70.0
            val target = -1.0 + 2.0 * x
            optimizer.observe(Vector(x), Vector(target))
        }

        assertApproxEquals(-1.0, network.evaluate(Vector(0))[0], 0.01)
        assertApproxEquals(3.0, network.evaluate(Vector(2))[0], 0.01)
        assertApproxEquals(9.0, network.evaluate(Vector(5))[0], 0.01)
    }

    @Test
    fun `calculates weights correctly with a constant`() {
        val rand = Random(0)
        val network = FeedforwardNeuralNetwork(
            stepSize = 0.0013,
            layers = arrayOf(
                NeuralLayer(
                    neuronCount = 3,
                    activationFunction = LogisticFunction,
                    regressorCount = 3,
                    sampler = NormalSampler(stdev = 2.0, random = rand)),
                NeuralLayer(
                    neuronCount = 3,
                    activationFunction = Tanh,
                    regressorCount = 3,
                    sampler = NormalSampler(stdev = 2.0, random = rand)),
                NeuralLayer(
                    neuronCount = 2,
                    activationFunction = IdentityFunction,
                    regressorCount = 3,
                    sampler = NormalSampler(stdev = 2.0, random = rand))
            )
        )

        val optimizer = AdamBackpropagation(
            network = network,
            costFunction = MeanCost(HalfSquaredError)
        )

        // out1(x1, x2, x3) = 1.5 + x1 + 2 * x2 + 0.5 * x3^2
        // out2(x1, x2, x3) = x1 * x2 + x3
        (-14..14).shuffled(rand).forEach { intX1 ->
            (-14..14).shuffled(rand).forEach { intX2 ->
                (-14..14).shuffled(rand).forEach { intX3 ->
                    val x1 = intX1.toDouble() / 12.0
                    val x2 = intX2.toDouble() / 12.0
                    val x3 = intX3.toDouble() / 12.0
                    val target1 = 1.5 + x1 + 2.0 * x2 + 0.5 * x3.pow(2)
                    val target2 = x1.pow(2) + x2 - x3
                    optimizer.observe(Vector(x1, x2, x3), Vector(target1, target2))
                }
            }
        }

        val estimate = network.evaluate(Vector(0.5, 1.5, 1.0))
        assertApproxEquals(5.5, estimate[0], 0.6)
        assertApproxEquals(0.75, estimate[1], 0.1)
    }

    @Test
    fun `backpropagates with softmax correctly`() {
        val rand = Random(0)
        val network = FeedforwardNeuralNetwork(
            stepSize = 0.01,
            layers = arrayOf(
                NeuralLayer(
                    neuronCount = 2,
                    activationFunction = Tanh,
                    regressorCount = 2,
                    sampler = NormalSampler(random = rand)),
                NeuralLayer(
                    neuronCount = 2,
                    activationFunction = Tanh,
                    regressorCount = 2,
                    sampler = NormalSampler(random = rand)),
                NeuralLayer.softmax(2)
            )
        )

        val optimizer = AdamBackpropagation(
            network = network,
            costFunction = SumCost(HalfSquaredError)
        )

        (0..100).shuffled(rand).forEach { intX1 ->
            (0..100).shuffled(rand).forEach { intX2 ->
                val x1 = intX1.toDouble() / 100.0
                val x2 = intX2.toDouble() / 100.0
                val target =
                    if (x1 < x2)
                        Vector(1, 0)
                    else
                        Vector(0, 1)
                optimizer.observe(Vector(x1, x2), target)
            }
        }

        val evaluateLessThan = network.evaluate(Vector(0.2, 0.8))
        assertTrue(evaluateLessThan[0] > 0.85)
        assertTrue(evaluateLessThan[0] + evaluateLessThan[1] == 1.0)

        val evaluateGreaterThan = network.evaluate(Vector(0.8, 0.2))
        assertTrue(evaluateGreaterThan[1] > 0.85)
        assertTrue(evaluateGreaterThan[0] + evaluateGreaterThan[1] == 1.0)

        // Test observeAndEvalute().
        val prevWeights = network.layers.first().neurons.first().weights.coeffs.copy()
        val observeAndEvaluate = optimizer.observeAndEvaluate(Vector(0.8, 0.2), Vector(0.9, 0.1))
        assertTrue(observeAndEvaluate[1] > 0.85)
        assertTrue(observeAndEvaluate[0] + observeAndEvaluate[1] == 1.0)
        assertNotEquals(prevWeights, network.layers.first().neurons.first().weights.coeffs)
    }

    @Test
    fun `batchObserveAndEvaluate() updates weights after batch`() {
        val network = FeedforwardNeuralNetwork(
            stepSize = 0.1,
            layers = arrayOf(
                NeuralLayer(
                    neuronCount = 1,
                    activationFunction = IdentityFunction,
                    regressorCount = 1,
                    includeConstant = false,
                    sampler = UniformSampler(1.0))
            )
        )
        val optimizer = AdamBackpropagation(
            network = network,
            costFunction = SumCost(HalfSquaredError)
        )

        assertApproxEquals(
            Vector(Vector(1), Vector(2), Vector(3)),
            optimizer.batchObserveAndEvaluate(
                Vector(Vector(1), Vector(2), Vector(3)),
                Vector(Vector(2), Vector(4), Vector(6))
            )
        )
        assertEquals(
            Vector(Vector(1.2680597213646203), Vector(2.5361194427292406), Vector(3.8041791640938607)),
            optimizer.batchObserveAndEvaluate(
                Vector(Vector(1), Vector(2), Vector(3)),
                Vector(Vector(2), Vector(4), Vector(6))
            )
        )
    }
}
