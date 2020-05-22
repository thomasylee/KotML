package kotml.regression.neural.initialization

import kotml.distributions.NormalSampler
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class HeInitializerTest {
    @Test
    fun `sampler() returns NormalSampler with correct mean and stdev`() {
        val sampler = HeInitializer.sampler(numInputs = 8, numOutputs = 4)
        assertTrue(sampler is NormalSampler)

        val normalSampler = sampler as NormalSampler
        assertEquals(0.0, normalSampler.mean)
        assertEquals(0.5, normalSampler.stdev)
    }
}
