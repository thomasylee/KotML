package kotml.reinforcement.replay

import kotlin.random.Random
import kotml.math.Vector
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class ExperienceReplayBufferTest {
    @Test
    fun `sample() randomly samples from replay buffer without duplicates`() {
        val random = Random(0)
        val buffer = ExperienceReplayBuffer(maxSize = 4)
        buffer.append(Vector(0), 0, 0.0, Vector(0))
        buffer.append(Vector(1), 0, 0.0, Vector(0))
        buffer.append(Vector(2), 0, 0.0, Vector(0))
        buffer.append(Vector(3), 0, 0.0, Vector(0))
        buffer.append(Vector(4), 0, 0.0, Vector(0))

        assertEquals(4, buffer.size)
        assertEquals(
            listOf(
                ExperienceReplayBuffer.Experience(
                    Vector(3), 0, 0.0, Vector(0)
                )
            ),
            buffer.sample(random = random)
        )
        assertEquals(
            listOf(
                ExperienceReplayBuffer.Experience(
                    Vector(2), 0, 0.0, Vector(0)
                ),
                ExperienceReplayBuffer.Experience(
                    Vector(1), 0, 0.0, Vector(0)
                ),
                ExperienceReplayBuffer.Experience(
                    Vector(4), 0, 0.0, Vector(0)
                )
            ),
            buffer.sample(3, random)
        )
        assertEquals(
            listOf(
                ExperienceReplayBuffer.Experience(
                    Vector(2), 0, 0.0, Vector(0)
                ),
                ExperienceReplayBuffer.Experience(
                    Vector(1), 0, 0.0, Vector(0)
                ),
                ExperienceReplayBuffer.Experience(
                    Vector(4), 0, 0.0, Vector(0)
                ),
                ExperienceReplayBuffer.Experience(
                    Vector(3), 0, 0.0, Vector(0)
                )
            ),
            buffer.sample(5, random)
        )
    }
}
