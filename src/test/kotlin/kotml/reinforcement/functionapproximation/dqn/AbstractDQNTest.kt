package kotml.reinforcement.functionapproximation.dqn

import kotlin.random.Random
import kotml.math.Vector
import kotml.regression.functions.IdentityFunction
import kotml.regression.functions.ReLU
import kotml.regression.neural.DenseNeuralLayer
import kotml.regression.neural.SplitNeuralLayer
// import kotml.regression.optimization.backpropagation.AdamBackpropagation
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class AbstractDQNTest {
    @Test
    fun `createNeuralNetwork() creates neural network correctly`() {
        val network = AbstractDQN.createNeuralNetwork(
            stateDimensions = 4,
            numActions = 2,
            random = Random(0)
        )
        // Make sure the numActions form returns the same network as the
        // layerSizes form with equivalent parameters.
        assertEquals(
            AbstractDQN.createNeuralNetwork(
                stateDimensions = 4,
                layerSizes = intArrayOf(4, 2),
                random = Random(0)
            ),
            network
        )

        assertEquals(2, network.layers.size)
        assertEquals(DenseNeuralLayer::class, network.layers[0]::class)
        assertEquals(DenseNeuralLayer::class, network.layers[1]::class)

        val firstNeuron = (network.layers[0] as DenseNeuralLayer).neurons.first()
        val lastNeuron = (network.layers[1] as DenseNeuralLayer).neurons.last()
        assertTrue(
            firstNeuron.activationFunction is ReLU,
            "First neuron should use ReLU"
        )
        assertTrue(
            lastNeuron.activationFunction is IdentityFunction,
            "Last neuron should use identity function"
        )
        assertEquals(2, network.evaluate(Vector(1, 2, 3, 4)).shape[0])
    }

    @Test
    fun `createDuelingNeuralNetwork() creates dueling neural network`() {
        val network = AbstractDQN.createDuelingNeuralNetwork(
            stateDimensions = 4,
            numActions = 2,
            commonLayerSizes = intArrayOf(4, 2),
            stateValueLayerSizes = intArrayOf(2, 1),
            advantageValueLayerSizes = intArrayOf(2, 2),
            random = Random(0)
        )

        assertEquals(4, network.layers.size)
        assertEquals(DenseNeuralLayer::class, network.layers[0]::class)
        assertEquals(DenseNeuralLayer::class, network.layers[1]::class)
        assertEquals(SplitNeuralLayer::class, network.layers[2]::class)
        assertEquals(DenseNeuralLayer::class, network.layers[3]::class)

        val firstCommonNeuron = (network.layers[0] as DenseNeuralLayer).neurons.first()
        val lastCommonNeuron = (network.layers[1] as DenseNeuralLayer).neurons.last()
        val firstStateNeuron = (
            (network.layers[2] as SplitNeuralLayer).subLayers[0][0] as DenseNeuralLayer
        ).neurons.first()
        val lastStateNeuron = (
            (network.layers[2] as SplitNeuralLayer).subLayers[0][1] as DenseNeuralLayer
        ).neurons.last()
        val firstAdvNeuron = (
            (network.layers[2] as SplitNeuralLayer).subLayers[1][0] as DenseNeuralLayer
        ).neurons.first()
        val lastAdvNeuron = (
            (network.layers[2] as SplitNeuralLayer).subLayers[1][1] as DenseNeuralLayer
        ).neurons.last()
        val aggregatingNeuron = (network.layers[3] as DenseNeuralLayer).neurons.first()

        assertEquals(ReLU::class, firstCommonNeuron.activationFunction::class)
        assertEquals(ReLU::class, lastCommonNeuron.activationFunction::class)
        assertEquals(ReLU::class, firstStateNeuron.activationFunction::class)
        assertEquals(IdentityFunction::class, lastStateNeuron.activationFunction::class)
        assertEquals(ReLU::class, firstAdvNeuron.activationFunction::class)
        assertEquals(IdentityFunction::class, lastAdvNeuron.activationFunction::class)
        assertEquals(IdentityFunction::class, aggregatingNeuron.activationFunction::class)
        assertEquals(
            DuelingAggregationFunction::class,
            aggregatingNeuron.aggregationFunction::class
        )
        assertEquals(2, network.evaluate(Vector(1, 2, 3, 4)).shape[0])
    }
}
