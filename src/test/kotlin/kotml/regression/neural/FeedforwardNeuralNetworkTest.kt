package kotml.regression.neural

import kotml.math.Vector
import kotml.regression.Weights
import kotml.regression.functions.IdentityFunction
import kotml.regression.functions.ReLU
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class FeedforwardNeuralNetworkTest {
    @Test
    fun `prevDenseLayers and nextDenseLayers have correct layer mappings`() {
        val firstLayer = DenseNeuralLayer(listOf(
            Neuron(
                activationFunction = IdentityFunction,
                weights = Weights(Vector(2))
            )
        ))
        val secondLayer = SplitNeuralLayer(listOf(
            listOf(
                DenseNeuralLayer(listOf(
                    Neuron(
                        activationFunction = IdentityFunction,
                        weights = Weights(Vector(3))
                    )
                ))
            ),
            listOf(
                DenseNeuralLayer(listOf(
                    Neuron(
                        activationFunction = IdentityFunction,
                        weights = Weights(Vector(4))
                    )
                ))
            )
        ))
        val thirdLayer = DenseNeuralLayer(listOf(
            Neuron(
                activationFunction = IdentityFunction,
                weights = Weights(Vector(1, 5))
            )
        ))
        val network = FeedforwardNeuralNetwork(listOf(
            firstLayer, secondLayer, thirdLayer
        ))

        // Number of inputs and outputs should be correct for each layer.
        assertEquals(1, firstLayer.numInputs)
        assertEquals(1, firstLayer.numOutputs)
        assertEquals(1, secondLayer.numInputs)
        assertEquals(2, secondLayer.numOutputs)
        assertEquals(2, thirdLayer.numInputs)
        assertEquals(1, thirdLayer.numOutputs)
        assertEquals(Vector(46), network.evaluate(Vector(1)))

        // Layers should be in order.
        assertEquals(
            listOf(firstLayer, secondLayer.subLayers[0][0] as DenseNeuralLayer,
                secondLayer.subLayers[1][0] as DenseNeuralLayer, thirdLayer),
            network.denseLayers
        )

        // Ensure that prev and next layer mappings are correct.
        assertTrue(
            network.prevDenseLayers[firstLayer].isNullOrEmpty(),
            "Prev dense layer for first layer should be null or empty"
        )
        assertEquals(
            listOf(secondLayer.subLayers[0][0], secondLayer.subLayers[1][0]),
            network.nextDenseLayers[firstLayer])
        assertEquals(
            listOf(firstLayer),
            network.prevDenseLayers[secondLayer.subLayers[0][0] as DenseNeuralLayer]
        )
        assertEquals(
            listOf(firstLayer),
            network.prevDenseLayers[secondLayer.subLayers[1][0] as DenseNeuralLayer]
        )
        assertEquals(
            listOf(thirdLayer),
            network.nextDenseLayers[secondLayer.subLayers[0][0] as DenseNeuralLayer]
        )
        assertEquals(
            listOf(thirdLayer),
            network.nextDenseLayers[secondLayer.subLayers[1][0] as DenseNeuralLayer]
        )
        assertEquals(
            secondLayer.subLayers.flatten(),
            network.prevDenseLayers[thirdLayer]
        )
        assertTrue(
            network.nextDenseLayers[thirdLayer].isNullOrEmpty(),
            "Next dense layer for last layer should be null or empty"
        )
    }

    @Test
    fun `evaluate() returns the correct output values`() {
        val hiddenLayer = DenseNeuralLayer((0..1).map { index ->
            Neuron(
                activationFunction = ReLU,
                weights = Weights(index.toDouble(), Vector(-0.25))
            )
        })
        val outputLayer = DenseNeuralLayer((0..1).map { index ->
            Neuron(
                activationFunction = ReLU,
                weights = Weights(index.toDouble(), Vector(2.0, -0.5))
            )
        })
        val network = FeedforwardNeuralNetwork(
            listOf(hiddenLayer, outputLayer)
        )
        // Input: [2]
        // Hidden layer: [0 - 0.25 * 2 => 0, 1 - 0.25 * 2 => 0.5]
        // Output layer: [0 + 2*0 - 0.5*0.5 => 0, 1 + 2*0 - 0.5*0.5 => 0.75]
        assertEquals(Vector(0.0, 0.75), network.evaluate(Vector(2)))
    }

    @Test
    fun `copy() returns a copy of the neural network`() {
        val original = FeedforwardNeuralNetwork(1, intArrayOf(1), ReLU)
        var copy = original.copy()
        assertEquals(original, copy)
        (original.layers.first() as DenseNeuralLayer).neurons.first().weights.coeffs[0] += 1.0
        assertNotEquals(original, copy)
    }

    @Test
    fun `updateWeights() updates the weights to the network's weights`() {
        val network = FeedforwardNeuralNetwork(listOf(
            DenseNeuralLayer(listOf(
                Neuron(
                    activationFunction = ReLU,
                    weights = Weights(1.0, Vector(2, 3))
                )
            ))
        ))
        val newWeights = Weights(5.0, Vector(6, 7))
        network.updateWeights(FeedforwardNeuralNetwork(listOf(
            DenseNeuralLayer(listOf(
                Neuron(
                    activationFunction = ReLU,
                    weights = newWeights
                )
            ))
        )))
        assertEquals(
            newWeights,
            (network.layers.first() as DenseNeuralLayer).neurons.first().weights
        )
    }
}
