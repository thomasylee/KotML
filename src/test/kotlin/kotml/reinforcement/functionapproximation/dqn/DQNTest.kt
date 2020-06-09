package kotml.reinforcement.functionapproximation.dqn

import kotlin.random.Random
import kotml.TestUtils.assertApproxEquals
import kotml.distributions.UniformSampler
import kotml.math.Vector
import kotml.regression.functions.IdentityFunction
import kotml.regression.neural.DenseNeuralLayer
import kotml.regression.neural.FeedforwardNeuralNetwork
import kotml.regression.neural.Neuron
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class DQNTest {
    @Test
    fun `learns and evaluates using the correct networks`() {
        val random = Random(0)
        val dqn = DQN(
            network = FeedforwardNeuralNetwork(listOf(
                DenseNeuralLayer(listOf(
                    Neuron(
                        activationFunction = IdentityFunction,
                        regressorCount = 1,
                        includeConstant = false,
                        sampler = UniformSampler(0.0)
                    )
                ))
            )),
            discount = 1.0,
            stepSize = 0.1,
            targetNetworkUpdateFrequency = 3,
            minibatchSize = 2,
            random = random
        )
        dqn.learn(Vector(1), 0, 1.0, null)

        assertEquals(Vector(0), dqn.targetNetwork.evaluate(Vector(1)))
        assertApproxEquals(Vector(0.1), dqn.network.evaluate(Vector(1)), 0.0001)
        assertApproxEquals(Vector(0.1), dqn.evaluate(Vector(1)), 0.0001)

        dqn.learn(Vector(1), 0, 2.0, null)
        assertApproxEquals(Vector(0.2968320098527678), dqn.targetNetwork.evaluate(Vector(1)))
        assertApproxEquals(Vector(0.2968320098527678), dqn.network.evaluate(Vector(1)))
        assertApproxEquals(Vector(0.2968320098527678), dqn.evaluate(Vector(1)))
    }
}
