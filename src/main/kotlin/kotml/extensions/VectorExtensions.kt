package kotml.extensions

import kotml.math.Vector

operator fun Double.plus(vector: Vector): Vector = vector.map { value ->
    this + value
}

operator fun Int.plus(vector: Vector): Vector = vector.map { value ->
    this + value
}

operator fun Double.minus(vector: Vector): Vector = vector.map { value ->
    this - value
}

operator fun Int.minus(vector: Vector): Vector = vector.map { value ->
    this - value
}

operator fun Double.times(vector: Vector): Vector = vector.map { value ->
    this * value
}

operator fun Int.times(vector: Vector): Vector = vector.map { value ->
    this * value
}

operator fun Double.div(vector: Vector): Vector = vector.map { value ->
    this / value
}

operator fun Int.div(vector: Vector): Vector = vector.map { value ->
    this / value
}
