package kotml.reinforcement.replay

import kotlin.random.Random
import kotml.math.Vector

/**
 * `ExperienceReplayBuffer` stores experiences and allows for random
 * sampling of the experiences in batches.
 */
class ExperienceReplayBuffer(val maxSize: Int = 1_000_000) {
    val experiences = mutableListOf<Experience>()
    var size: Int = 0

    data class Experience(
        val state: Vector,
        val action: Int,
        val reward: Double,
        val nextState: Vector,
        val isTerminal: Boolean
    )

    /**
     * Adds the experience to the replay buffer.
     * @param state previous state
     * @param action action taken in the previous state
     * @param reward reward from taking the action in the previous state
     * @param nextState next state after having take the action
     * @param isTerminal true if the nextState is a terminal state, false otherwise
     */
    fun append(state: Vector, action: Int, reward: Double, nextState: Vector, isTerminal: Boolean) {
        if (size == maxSize)
            experiences.removeAt(0)
        else
            size++
        experiences.add(Experience(state, action, reward, nextState, isTerminal))
    }

    /**
     * Returns a list of random experiences from the buffer. If the desired
     * number of experiences exceeds the number of experiences in the buffer,
     * the experiences in the buffer are returned in random order. A single
     * call to sample() does not return duplicates unless the same experience
     * is in the buffer multiple times.
     * @param count number of experiences to return
     * @param random source of randomness
     * @return list of random experiences
     */
    fun sample(count: Int = 1, random: Random = Random): List<Experience> =
        if (count >= size) {
            experiences.shuffled(random)
        } else {
            val indices = (0 until size).map { it }.toMutableList()
            (1..count).map {
                val indicesIndex = random.nextInt(0, indices.size)
                val experienceIndex = indices[indicesIndex]
                indices.removeAt(indicesIndex)
                experiences[experienceIndex]
            }
        }
}
