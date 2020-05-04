package kotml.distributions

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class UniformSamplerTest {
    @Test
    fun `sample() returns 0 when no value is provided in the constructor`() {
        assertEquals(0.0, UniformSampler().sample())
    }

    @Test
    fun `sample() returns the value provided in the constructor`() {
        assertEquals(1.0, UniformSampler(1.0).sample())
    }
}
