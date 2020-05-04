package kotml.distributions

class UniformSampler(val value: Double = 0.0) : DistributionSampler {
    override fun sample(): Double = value
}
