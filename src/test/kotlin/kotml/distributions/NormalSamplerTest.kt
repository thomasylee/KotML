package kotml.distributions

import kotlin.random.Random
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class NormalSamplerTest {
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
        // Make 5 runs with different random seeds.
        (1..5).forEach {
            val sampler = NormalSampler(random = Random(it))
            assertTrue(evaluateSamples(sampler))
        }
    }

    @Test
    fun `sample() uses mean and stdev`() {
        // Make 5 runs with different random seeds.
        (1..5).forEach {
            val sampler = NormalSampler(mean = 100.0, stdev = 20.0, random = Random(it))
            assertTrue(sampler.sample() > 20.0 && evaluateSamples(sampler))
        }
    }
}
