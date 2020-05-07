package kotml.distributions

import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class NormalDistributionTest {
    private fun evaluateSamples(sampler: NormalSampler): Boolean {
        val samples = (1..10).map { sampler.sample() }
        val (inStdev, outStdev) = samples.partition { sample ->
            sample - sampler.mean >= -sampler.stdev &&
                sample - sampler.mean <= sampler.stdev
        }
        // 96.4% chance >= 5 samples are within one stdev and >= 1 samples
        // are outside one stdev.
        return inStdev.size >= 5 && outStdev.size >= 1
    }

    @Test
    fun `sample() returns normally distributed samples`() {
        val sampler = NormalSampler()
        // Try up to 3 times to take 10 samples and evaluate distribution.
        (1..3).forEach {
            if (evaluateSamples(sampler)) return
        }
        assertTrue(false)
    }

    @Test
    fun `sample() uses mean and stdev`() {
        val sampler = NormalSampler(mean = 100.0, stdev = 20.0)
        // Try up to 3 times to take 10 samples and evaluate distribution.
        (1..3).forEach {
            if (sampler.sample() > 20.0 && evaluateSamples(sampler))
                return
        }
        assertTrue(false)
    }
}
