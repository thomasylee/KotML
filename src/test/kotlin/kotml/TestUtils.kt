package kotml

import kotlin.math.abs
import kotml.math.Vector
import org.opentest4j.AssertionFailedError

object TestUtils {
    const val defaultTolerance = 0.00000001

    fun assertApproxEquals(expected: Vector, actual: Vector, tolerance: Double = defaultTolerance) {
        if (!actual.approxEquals(expected, tolerance))
            throw AssertionFailedError("Expected $actual to approxEqual $expected")
    }

    fun assertApproxEquals(expected: Double, actual: Double, tolerance: Double = defaultTolerance) {
        if (abs(actual - expected) >= tolerance)
            throw AssertionFailedError("Expected $actual to approxEqual $expected")
    }
}
