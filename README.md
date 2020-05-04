# KotML
Kotlin Machine Learning

## Vectors

Use a `Vector` instance to represent an n-dimensional array of values, similarly to in numpy's array class.
```kotlin
import kotml.math.Vector

// Initialize with vararg Double values.
Vector(-1.0, 0.0, 1.0)

// Int values get converted to Double.
Vector(1, 2) == Vector(1.0, 2.0)

// Create vectors of vectors to express multiple dimensions.
Vector(Vector(1, 2), Vector(3, 4), Vector(5, 6)) // [[1,2], [3,4], [5,6]]

// You can create a vector of zeros with any valid shape.
Vector.zeros(intArray(2, 4)) == Vector(Vector(0, 0, 0, 0), Vector(0, 0, 0, 0))

// You can even create vectors with values determined by index.
Vector(intArray(2, 3)) { 2 * it } == Vector(Vector(0, 2, 4), Vector(6, 8, 10))
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

// Matrix multiplication
vector x vector.transpose() == Vector(85)
```

Vectors also support mapping, as well as folding, adding, and multiplying along an axis, similarly to in numpy.
```kotlin
Vector(Vector(1, 2), Vector(3, 4)).map { it * 2 } ==
    Vector(Vector(2, 4), Vector(6, 8))

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

## Regression

A `FunctionModel` represents a mathemtical function such as a polynomial (`Polynomial` class) or logistic function (`LogisticFunction` object).

A `CostFunction` represents a differentiable cost function such as least squared error (`OrdinaryLeastSquares` object).

You can use `StochasticGradientDescent` to estimate functions by building weights that minimize cost.
```kotlin
import kotml.math.Vector
import kotml.regression.functions.Polynomial
import kotml.regression.objectives.OrdinaryLeastSquares
import kotml.regression.optimization.StochasticGradientDescent

// Use stochastic gradient descent on a polynomial of the
// form a + b * x^2, with ordinary least squares cost.
val estimator = StochasticGradientDescent(
    stepSize = 0.001,
    regressorCount = 1,
    function = Polynomial(Vector(2.0)), // The only regressor has exponent = 2.
    costFunction = OrdinaryLeastSquares)

// This should look similar to y = x^2.
estimator.addObservation(24.8, Vector(5.0))
estimator.addObservation(9.1, Vector(3.0))
estimator.addObservation(-0.1, Vector(0.0))
estimator.addObservation(3.7, Vector(-2.0))
estimator.addObservation(8.8, Vector(-3.0))
estimator.addObservation(16.1, Vector(-4.0))

val twoSquaredEstimate = estimator.estimate(Vector(2.0))
(10.0 * twoSquaredEstimate).toInt() / 10.0 == 4.3
```
