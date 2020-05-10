# KotML
Kotlin Machine Learning

## Vectors

Use a `Vector` instance to represent an n-dimensional array of values, similarly to in numpy's array class.
```kotlin
import kotml.math.Vector

// Initialize with vararg Double or Int values.
Vector(-1.0, 0.0, 1.0)

// Int values get converted to Double.
Vector(1, 2) == Vector(1.0, 2.0)

// Create vectors of vectors to express multiple dimensions.
Vector(Vector(1, 2), Vector(3, 4), Vector(5, 6)) // [[1,2], [3,4], [5,6]]

// You can create a vector of zeros with any valid shape.
// Note that the 2, 4 arguments here are the shape of the vector.
Vector.zeros(2, 4) == Vector(Vector(0, 0, 0, 0), Vector(0, 0, 0, 0))

// You can even create vectors with values determined by index.
// Note that the 2, 3 arguments here are the shape of the vector.
Vector(2, 3) { 2 * it } == Vector(Vector(0, 2, 4), Vector(6, 8, 10))
```

Due to Kotlin's static typing, we use `[]` to access `Vector` elements, and `()` to access `Double` elements.
```kotlin
val vector = Vector(
    Vector(
        Vector(1, 2, 3),
        Vector(4, 5, 6)),
    Vector(
        Vector(7, 8, 9),
        Vector(10, 11, 12)))

vector(0) == Vector(Vector(1, 2, 3), Vector(4, 5, 6))
vector(0, 1) == Vector(4, 5, 6)
vector[0, 0, 0] == 1.0
vector[0, 1, 1] == 5.0
vector[1, 1, 2] == 12.0
```

Vectors support basic mathematical operations.
```kotlin
import kotml.extensions.plus // Provides 1 + vector for Double and Int
import kotml.extensions.minus // Provides 1 - vector for Double and Int
import kotml.extensions.times // Provides 1 * vector for Double and Int
import kotml.extensions.div // Provides 1 / vector for Double and Int
import kotml.math.Vector

val vector = Vector(2, 9)

vector + 1 == Vector(3, 10)
vector - 1 == Vector(1, 8)
vector * 2 == Vector(4, 18)
vector / 2 == Vector(1.0, 4.5)

vector + vector == Vector(4, 18)
vector - vector == Vector(0, 0)
vector * vector == Vector(4, 81)
vector / vector == Vector(1, 1)

// Dot product
vector dot vector == 85.0

// Transpose
vector.transpose() == Vector(Vector(2), Vector(9))

// Matrix multiplication
vector x vector.transpose() == Vector(85)

// Determinant
Vector(2, 2) { it }.det() == -2.0
```

Vectors also support mapping, as well as folding, adding, and multiplying along an axis, similarly to in numpy.
```kotlin
Vector(Vector(1, 2), Vector(3, 4)).map { it * 2 } ==
    Vector(Vector(2, 4), Vector(6, 8))

Vector(1, 2, 3).mapIndexed { index, value -> index * value } == Vector(0, 2, 6)

// Setting axis = 0 is the default.
Vector(Vector(1), Vector(2), Vector(3)).fold(initial = 0.0, axis = 0) { acc, value ->
    acc + 2 * value
} == Vector(12.0)

Vector(Vector(1), Vector(2), Vector(3)).fold(initial = 0.0, axis = 1) { acc, value ->
    acc + 2 * value
} == Vector(2, 4, 6)

Vector(1, 2).sum() == Vector(3)
Vector(Vector(1), Vector(2)).sum(axis = 1) == Vector(1, 2)

Vector(1, 2).product() == Vector(2)
Vector(Vector(1), Vector(2)).product(axis = 1) == Vector(1, 2)
```

The usual `Vector` class is immutable, but you can use `MutableVector` to create vectors whose values can be updated. Note that the shape of the vector cannot change, but you can reassign subvector and scalar entries.
```kotlin
// You can also use Vector(Vector(1, 2), Vector(3, 4)).toMutableVector().
// Note that the Vector arguments also get converted to MutableVectors.
val mutable = MutableVector(Vector(1, 2), Vector(3, 4))

mutable[0] = Vector(-1, -2)
mutable[1] = MutableVector(-3, -4)
mutable == MutableVector(Vector(-1, -2), Vector(-3, -4))

mutable[0, 0] = 5
mutable[0, 1] = 6.0
mutable(0) == Vector(5, 6)
```

## Regression

A `FunctionModel` represents a mathemtical function such as a polynomial (`Polynomial` class) or logistic function (`LogisticFunction` object).

A `CostFunction` represents a differentiable cost function such as least squared error (`OrdinaryLeastSquares` object).

`Weights` tracks the coefficient weights and, optionally, a constant.
```kotlin
Weights(0.1, doubleArray(0.2, 0.3)) // constant = 0.1, coeffs = 0.2, 0.3
Weights(doubleArray(0.2, 0.3)) // no constant, coeffs = 0.2, 0.3

Weights(3, true) // constant = 0, coeffs = 0, 0, 0
Weights(3) // no constant, coeffs = 0, 0, 0

val weights = Weights(2, true, UniformSampler(1.0))
weights.hasConstant == true
weights.constant == 1.0
weights.coeffs[0] == 1.0
weights.coeffs[1] == 1.0
```

You can use a `WeightedOptimizer` like `StochasticGradientDescent` to build models of functions by selecting weights that minimize a cost function.
```kotlin
import kotml.math.Vector
import kotml.regression.cost.loss.SquaredError
import kotml.regression.functions.Polynomial
import kotml.regression.optimization.StochasticGradientDescent

// Use stochastic gradient descent on a polynomial of the
// form a + b * x^2, with ordinary least squares cost.
val estimator = StochasticGradientDescent(
    stepSize = 0.001,
    function = Polynomial(Vector(2.0)), // The only regressor has exponent = 2.
    lossFunction = SquaredError,
    regressorCount = 1
)

// This should look similar to y = x^2.
estimator.addObservation(24.8, Vector(5.0))
estimator.addObservation(9.1, Vector(3.0))
estimator.addObservation(-0.1, Vector(0.0))
estimator.addObservation(3.7, Vector(-2.0))
estimator.addObservation(8.8, Vector(-3.0))
estimator.addObservation(16.1, Vector(-4.0))

val twoSquaredEstimate = estimator.function.evaluate(estimator.weights, Vector(2.0))
(10.0 * twoSquaredEstimate).toInt() / 10.0 == 4.3
```
