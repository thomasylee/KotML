package kotml.distributions

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * NormalSampler samples random values from a normal distribution using
 * the Box–Muller transform.
 *
 * TODO: Switch to the more efficient Ziggurat algorithm.
 */
class NormalSampler(
    val mean: Double = 0.0,
    val stdev: Double = 1.0
) : DistributionSampler {
    private var useCosine: Boolean = true
    private lateinit var rand: Pair<Double, Double>

    override fun sample(): Double =
        if (useCosine) {
            useCosine = false
            rand = Pair(Random.nextDouble(), Random.nextDouble())
            sqrt(-2.0 * ln(rand.first)) * cos(2.0 * PI * rand.second) * stdev + mean
        } else {
            useCosine = true
            sqrt(-2.0 * ln(rand.first)) * sin(2.0 * PI * rand.second) * stdev + mean
        }
}
