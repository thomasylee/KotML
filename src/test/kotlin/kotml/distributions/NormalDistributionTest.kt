package kotml.distributions

import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class NormalDistributionTest {
    private fun evaluateSamples(): Boolean {
        val sampler = NormalSampler()
        val samples = (1..10).map { sampler.sample() }
        val (inStdev, outStdev) = samples.partition { it >= -1.0 && it <= 1.0 }
        // 96.4% chance >= 5 samples are within one stdev and >= 1 samples
        // are outside one stdev.
        return inStdev.size >= 5 && outStdev.size >= 1
    }

    @Test
    fun `sample() returns normally distributed samples`() {
        // Try up to 3 times to take 10 samples and evaluate distribution.
        (1..3).forEach {
            if (evaluateSamples()) return
        }
        assertTrue(false)
    }
}
