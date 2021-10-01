package com.linkedin.dualip.projection

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, SparseVector => BSV}
import com.linkedin.dualip.blas.VectorOperations._
import scala.collection.mutable


/**
  * The enumeration of available vertices
  */
object VertexType extends Enumeration {
  type VertexType = Value
  val Closest = Value("closest")
  val Farthest = Value("farthest")
}
import VertexType._

/**
 * Polytope projection where the polytope is represented by a set of points. maxIter bounds the number of iterations
 * of Wolfe's algorithm.
 *
 */
class PolytopeProjection(polytope: mutable.Set[BSV[Double]], maxIter: Int) extends Projection with Serializable {

  case class VertexSolution(firstBest: BSV[Double], nextBest: BSV[Double], isOptimal: Boolean)

  val tolerance = 10e-10

  /**
    * Pick the top-k elements from an iterable in O(Nlogk) time using a priority queue
    * @param k        - top k elements needed
    * @param iterable - sequence of elements
    * @param ord      - for complex types, we specify an ordering to help pick top k
    * @tparam T       - type of element in the sequence
    * @return
    */
  def pickTopN[T](k: Int, iterable: Iterable[T])(implicit ord: Ordering[T]): Seq[T] = {
    val q = collection.mutable.PriorityQueue[T](iterable.toSeq: _*)
    val end = Math.min(k, q.size)
    (1 to end).map(_ => q.dequeue())
  }

  /**
    * Given a vertex (v) that is not inside the polytope find the closest (max dot product) or farthest (min dot product)
    * vertex from v.
    * @param v          - vertex representing primal solution without any polytope constraints
    * @param vertexType - closest or farthest vertex
    * @return
    */
  def candidateVertex(v: BSV[Double], vertexType: VertexType, metadata: Metadata): (BSV[Double], Double) = {
    val ordering: Ordering[BSV[Double]] = vertexType match {
      case Closest => Ordering.by(vertex =>  dot(vertex, v))
      case Farthest => Ordering.by(vertex => -dot(vertex, v))
    }
    val vertex: BSV[Double] = pickTopN[BSV[Double]](1, polytope)(ordering).head
    (vertex, dot(vertex, v))
  }

  /**
    * Find the best vertex & check if the best vertex satisfies optimality. If it does, we return
    * it along with its distance from v. If not, we find the second best vertex and return its distance
    * from v
    * @param v - vertex representing primal solution without any polytope constraints
    * @return the top 2 vertices closest to v, along with a check for optimality
    */
  def checkVertexSolution(v: BSV[Double], metadata: Metadata): VertexSolution = {
    def getNearestVertex(v: BSV[Double]): BSV[Double] = {
      pickTopN[BSV[Double]](1, polytope)(Ordering.by(vertex =>  -dot(vertex - v, vertex - v))).head
    }

    val firstBest = getNearestVertex(v)
    polytope.remove(firstBest)

    val nextBest = getNearestVertex(v - firstBest)
    polytope.add(firstBest)

    val isOptimal = dot(v - firstBest, firstBest) > dot(v - firstBest, nextBest)

    VertexSolution(firstBest = firstBest, nextBest = nextBest, isOptimal = isOptimal)
  }

  /**
    * Find the min norm point in an affine hull along with its barycentric coordinates
    * i.e. find an x where min_x 0.5 ||Sλ||^2 such that e^Tλ = 1
    *
    * @param S  - sequence of points representing the affine hull
    * @return
    */
  def affineMinimizer(S: Array[BSV[Double]]): (BSV[Double], Array[Double]) = {
    val scores = BDM.ones[Double](S.length + 1, S.length + 1)
    // scores represents a matrix of dot products between all pairs of points
    scores(0, 0) = 0
    for (i <- S.indices) {
      for (j <- 0 to i) {
        // todo(rramanat): Optimize this in the future. We don't need to recompute all dot products on every iteration
        scores(i + 1, j + 1) = dot(S(i), S(j))
        scores(j + 1, i + 1) = scores(i + 1, j + 1)
      }
    }
    // The following definition of e also works. Leaving it in here to investigate for the future
    // val e = BDV.ones[Double](S.length + 1)
    val e = BDV.zeros[Double](S.length + 1)
    e(0) = 1
    val solution = scores \ e
    // Barycentric coordinates
    val lambdas = solution.data.slice(1, S.length + 1)
    // Since solution x = \sum_i \lambda_i * point_i
    val point = sum(
      lambdas.zipWithIndex.map { case (lambda, index) =>
        multiply(S(index), lambda)
      }
    )
    (point, lambdas)
  }

  /**
    * Minor cycle of Wolfe's algorithm
    * @param affineHull             - the sequence of points representing the affine hull
    * @param previousMinimizer      - the min norm solution found so far
    * @param barycentricCoordinates - corresponding to the min norm solution
    * @return
    */
  def minorCycle(affineHull: Seq[BSV[Double]],
    previousMinimizer: BSV[Double],
    barycentricCoordinates: Array[Double]): (Seq[BSV[Double]],BSV[Double], Array[Double]) = {
    // Making a copy of the variables since they will be re-assigned in the algorithm.
    var S: Seq[BSV[Double]] = affineHull
    var x: BSV[Double] = previousMinimizer
    var lambdas: Array[Double] = barycentricCoordinates

    var searchMinorCycle = true
    while(searchMinorCycle) {
      // y = arg min_{z ∈ aff(S)} ‖z‖
      val (y, alphas): (BSV[Double], Array[Double]) = affineMinimizer(S.toArray)
      if (alphas.forall(_ >= 0)) {  // If y ∈ conv(S), then end minor cycle
        x = y; lambdas = alphas
        searchMinorCycle = false
      } else {
        // If !(y ∈ conv(S)), then update x to the intersection of the boundary of conv(S) and the segment joining y
        // and previous x.  Delete points from S which are not required to describe the new x as a convex combination
        val (theta, thetaIndex): (Double, Int) = lambdas.zip(alphas).zipWithIndex
          .flatMap { case ((lambda, alpha), index) =>
            if (alpha < 0 && (lambda - alpha > 0)) {
            Some(lambda / (lambda - alpha), index)
            } else {
              None
            }
          }.min // Since x = ∑_i λ_i q_i
        // Using θ, the new x lies in conv(S).
        x = multiply(y, theta) + multiply(x, 1 - theta)
        // // Sets the coefficients of the new x
        lambdas = sum(multiply(alphas, theta), multiply(lambdas, 1 - theta))
        // Delete points which have λ_i = 0
        S.zip(lambdas).zipWithIndex.filter { case ((_, lambda), index) => index != thetaIndex && lambda > 0 }
          .map { case ((point, lambda), index) => (point, lambda)}
          .unzip match {
          case (newHull, newCoordinates) => S = newHull; lambdas = newCoordinates.toArray
          case _ => // Scala uses patten matching for multiple assignment https://stackoverflow.com/a/6197785
        }
      }
    }
    (S, x, lambdas)
  }

  /**
    * Check to detect if we have found the min norm solution. If it fails, the vertex can be used to reduce the min
    * norm solution further
    * @param solution  - the min norm solution found so far
    * @param vertex    - the next vertex we want to evaluate
    * @return
    */
  def checkOptimality(solution: BSV[Double], vertex: BSV[Double]): Boolean = {
    // s^Ts <= s^T v + tolerance * s^Ts
    dot(solution, solution) - dot(solution, vertex) <= tolerance * dot(solution, solution)
  }

  def wolfe(xHat: BSV[Double], firstBest: BSV[Double], nextBest: BSV[Double], metadata: Metadata): BSV[Double] = {
    // Making a copy of the variables since they will be re-assigned in the algorithm.
    var S = Array[BSV[Double]]()
    var lambdas = Array[Double]()

    // Shift the axis to find point of min distance from xHat instead of the min norm solution, i.e. min distance from origin
    var x = firstBest - xHat
    // We always maintain x=∑_{i∈S} λ_i q_i as a convex combination of a subset S of vertices of B.
    S = S :+ (firstBest - xHat)
    lambdas = lambdas :+ 1.0

    // If its the first major cycle, we have already found the candidate vertex in the check outside wolfe's algorithm,
    // we can reuse that vertex as the first candidate
    var firstMajorCycle = true

    for (iter <- 0 to maxIter) {
      var (q, _) = if (firstMajorCycle) {
        firstMajorCycle = false
        (nextBest, None)
      } else {
        // q ∈ arg min_{p ∈ B} xTp
        candidateVertex(x, VertexType.Farthest, metadata)
      }
      // candidate vertex can find the next vertex using the special structure of our problem. However, wolfe's algorithm
      // operates in a different coordinate system (shifted by xHat) where that structure is absent.
      q = q - xHat

      if (checkOptimality(solution = x, vertex = q)) {
        return x + xHat // this is the solution
      }
      S = S :+ q
      lambdas = lambdas :+ 0.0
      minorCycle(S, x, lambdas) match {
        // After the minor loop terminates, x is updated to be the affine minimizer of the current set S
        case (newCorral, newMinima, newCoordinates) => S = newCorral.toArray; x = newMinima; lambdas = newCoordinates
        case _ =>
      }
    }
    // If we have attempted to find the solution for maxIter major cycle and have not found, we return a sub-optimal value
    x + xHat
  }

  def equalityConstraint(xHat: BSV[Double], metadata: Metadata): BSV[Double] = {
    // If the closest vertex is the optimal solution, we are done
    val vertexSolution = checkVertexSolution(xHat, metadata)
    if (vertexSolution.isOptimal) {
      vertexSolution.firstBest
    } else {
      wolfe(xHat, vertexSolution.firstBest, vertexSolution.nextBest, metadata)
    }
  }

  /**
    * The entry point to project a point onto polytope contraints
    * @param xHat - solution to the primal that does not satisfy polytope constraints
    *  @return projected primal variable
    */
  override def project(xHat: BSV[Double], metadata: Metadata): BSV[Double] = {
    equalityConstraint(xHat, metadata)
  }
}