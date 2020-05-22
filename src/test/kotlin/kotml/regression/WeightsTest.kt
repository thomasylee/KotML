package kotml.regression

import kotml.distributions.UniformSampler
import kotml.math.Vector
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class WeightsTest {
    @Test
    fun `Weights(Int, Boolean, DistributionSampler) initializes correctly`() {
        val withConstant = Weights(2, true, UniformSampler(3.0))
        assertTrue(withConstant.hasConstant)
        assertEquals(3.0, withConstant.constant)
        assertEquals(2, withConstant.coeffs.shape[0])
        assertEquals(3.0, withConstant.coeffs[0])
        assertEquals(3.0, withConstant.coeffs[1])

        val withoutConstant = Weights(2, false, UniformSampler(3.0))
        assertFalse(withoutConstant.hasConstant)
        assertEquals(0.0, withoutConstant.constant)
        assertEquals(2, withoutConstant.coeffs.shape[0])
        assertEquals(3.0, withoutConstant.coeffs[0])
        assertEquals(3.0, withoutConstant.coeffs[1])
    }

    @Test
    fun `toVector() returns vector with coeffs followed by constant`() {
        val withConstant = Weights(3.0, Vector(1, 2))
        assertTrue(withConstant.hasConstant)
        assertEquals(Vector(1, 2, 3), withConstant.toVector())

        // Constant of 0 gets added when hasConstant is false.
        val withoutConstant = Weights(Vector(1, 2))
        assertFalse(withoutConstant.hasConstant)
        assertEquals(Vector(1, 2, 0), withoutConstant.toVector())
    }

    @Test
    fun `copy() returns a copy of the weights`() {
        val originalWithBias = Weights(2.0, Vector(3, 4))
        val copyWithBias = originalWithBias.copy()
        originalWithBias.constant = 3.0
        originalWithBias.coeffs[0] = 5.0
        assertEquals(Weights(3.0, Vector(5, 4)), originalWithBias)
        assertEquals(Weights(2.0, Vector(3, 4)), copyWithBias)

        val originalWithoutBias = Weights(Vector(3, 4))
        val copyWithoutBias = originalWithoutBias.copy()
        originalWithoutBias.coeffs[0] = 5.0
        assertEquals(Weights(Vector(5, 4)), originalWithoutBias)
        assertEquals(Weights(Vector(3, 4)), copyWithoutBias)
    }
}
