package kotml.reinforcement.functionapproximation

import kotlin.random.Random
import kotml.TestUtils.assertApproxEquals
import kotml.distributions.UniformSampler
import kotml.math.Vector
import kotml.regression.functions.IdentityFunction
import kotml.regression.neural.FeedforwardNeuralNetwork
import kotml.regression.neural.NeuralLayer
import kotml.regression.neural.Neuron
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class DeepQNetworkTest {
    @Test
    fun `learns and evaluates using the correct networks`() {
        val random = Random(0)
        val dqn = DeepQNetwork(
            network = FeedforwardNeuralNetwork(arrayOf(
                NeuralLayer(arrayOf(
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
            minibatchSize = 2
        )
        dqn.learn(Vector(1), 0, 1.0, Vector(0), false, random)

        assertEquals(Vector(0), dqn.targetNetwork.evaluate(Vector(1)))
        assertApproxEquals(Vector(0.1), dqn.network.evaluate(Vector(1)), 0.0001)
        assertApproxEquals(Vector(0.1), dqn.evaluate(Vector(1)), 0.0001)

        dqn.learn(Vector(1), 0, 2.0, Vector(0), false, random)
        assertApproxEquals(Vector(0.2888502494508579), dqn.targetNetwork.evaluate(Vector(1)), 0.001)
        assertApproxEquals(Vector(0.2888502494508579), dqn.network.evaluate(Vector(1)), 0.001)
        assertApproxEquals(Vector(0.2888502494508579), dqn.evaluate(Vector(1)), 0.001)
    }
}
