package kotml.distributions

interface DistributionSampler {
    /**
     * Returns a random sample according to a particular distribution.
     * @return random sample
     */
    abstract fun sample(): Double
}
