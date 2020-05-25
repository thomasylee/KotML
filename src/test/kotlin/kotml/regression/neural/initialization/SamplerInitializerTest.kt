package kotml.regression.neural.initialization

import kotml.distributions.NormalSampler
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class SamplerInitializerTest {
    @Test
    fun `sampler() returns the SamplerInitializer's distribution sampler`() {
        val sampler = NormalSampler(mean = 0.1, stdev = 0.5)
        assertEquals(sampler, SamplerInitializer(sampler).sampler(0, 0))
    }
}
