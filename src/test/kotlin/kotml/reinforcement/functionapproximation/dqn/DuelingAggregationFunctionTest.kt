package kotml.reinforcement.functionapproximation.dqn

import kotml.math.Vector
import kotml.regression.Weights
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class DuelingAggregationFunctionTest {
    @Test
    fun `meanAdvantage() returns the average advantage value`() {
        val function = DuelingAggregationFunction(0)
        assertEquals(4.5, function.meanAdvantage(Vector(-10, 3, 4, 5, 6)))
    }

    @Test
    fun `aggregate() returns the right action value estimate`() {
        val weights = Weights(Vector.zeros(1))
        val function1 = DuelingAggregationFunction(0)
        val function2 = DuelingAggregationFunction(1)
        val function3 = DuelingAggregationFunction(2)
        val regressors = Vector(-2, 0, 2, 7)
        assertEquals(-5.0, function1.aggregate(weights, regressors))
        assertEquals(-3.0, function2.aggregate(weights, regressors))
        assertEquals(2.0, function3.aggregate(weights, regressors))
    }

    @Test
    fun `weightsGradient() returns zeros`() {
        val weights = Weights(Vector(1, 2))
        val regressors = Vector(3, 4)
        val function = DuelingAggregationFunction(0)
        assertEquals(
            Weights(Vector(0, 0)),
            function.weightsGradient(weights, regressors)
        )
    }

    @Test
    fun `regressorsGradient() returns correct value`() {
        val weights = Weights(Vector(1, 2, 3, 4, 5))
        val regressors = Vector(6, 7, 8, 9, 10)
        val function = DuelingAggregationFunction(0)
        assertEquals(
            Vector(1.0, 0.75, 0.0, 0.0, 0.0),
            function.regressorsGradient(weights, regressors)
        )
    }
}
