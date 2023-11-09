package com.linkedin.dualip.projection

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.projection.PolytopeProjectionTest._
import com.linkedin.dualip.util.VectorOperations.dot
import org.scalatest.matchers.should.Matchers._
import org.testng.Assert
import org.testng.annotations.Test

import scala.collection.mutable
import scala.math.pow

class PolytopeProjectionTest {

  @Test
  def testVanillaAffineMinimizer(): Unit = {
    case class AffineMinimizerTest(affineHull:  Array[BSV[Double]], expectedMinima: BSV[Double], expectedCoordinates: Array[Double])
    val data: Seq[AffineMinimizerTest] = Seq(
      AffineMinimizerTest(
        affineHull = Array[BSV[Double]](
          new BSV[Double](Array(0, 1), Array(0, 2), 2),
          new BSV[Double](Array(0, 1), Array(3, 0), 2)),
        expectedMinima = new BSV[Double](Array(0, 1), Array(0.92, 1.38), 2),
        expectedCoordinates = Array(0.69, 0.31)
      ),
      AffineMinimizerTest(
        affineHull = Array[BSV[Double]](
          new BSV[Double](Array(0, 1), Array(3, 0), 2),
          new BSV[Double](Array(0, 1), Array(-2, 1), 2)),
        expectedMinima = new BSV[Double](Array(0, 1), Array(0.12, 0.58), 2),
        expectedCoordinates = Array(0.42, 0.58)
      )
    )

    data.foreach { point =>
      val polytopeProjection = new PolytopeProjection(null, 10)
      val (answer, lambdas) = polytopeProjection.affineMinimizer(point.affineHull)
      Assert.assertEquals(truncate(answer), point.expectedMinima)
      truncate(lambdas, 2).zip(point.expectedCoordinates).foreach { case (i, e) =>
        Assert.assertEquals(i, e)
      }
    }
  }

  @Test
  def testPolytopeProjectionOrigin(): Unit = {
    case class PolygonProjectionTest(polytope: mutable.Set[BSV[Double]], closestVertex: BSV[Double], nextBestVertex: BSV[Double], minima: BSV[Double])
    val data: Seq[PolygonProjectionTest] = Seq(
      // Example from http://essay.utwente.nl/81914/1/Petrov_BA_EEMCS.pdf
      PolygonProjectionTest(
        polytope = mutable.Set(
          new BSV[Double](Array(0, 1), Array(0, 2), 2),
          new BSV[Double](Array(0, 1), Array(3, 0), 2),
          new BSV[Double](Array(0, 1), Array(-2, 1), 2)),
        closestVertex = new BSV[Double](Array(0, 1), Array(0, 2), 2),
        nextBestVertex = new BSV[Double](Array(0, 1), Array(-2, 1), 2),
        minima = new BSV[Double](Array(0, 1), Array(0.12, 0.58), 2)
      ),
      // Example from https://arxiv.org/pdf/1710.02608.pdf
      PolygonProjectionTest(
        polytope = mutable.Set(
          new BSV[Double](Array(0, 1, 2), Array(0.8, 0.9, 0), 3),
          new BSV[Double](Array(0, 1, 2), Array(1.5, -0.5, 0), 3),
          new BSV[Double](Array(0, 1, 2), Array(-1.0, -1.0, 2.0), 3),
          new BSV[Double](Array(0, 1, 2), Array(-4.0, 1.5, 2.0), 3)),
        closestVertex = new BSV[Double](Array(0, 1, 2), Array(0.8, 0.9, 0), 3),
        nextBestVertex = new BSV[Double](Array(0, 1, 2),  Array(-1.0, -1.0, 2.0), 3),
        minima = new BSV[Double](Array(0, 1, 2), Array(0.20, 0.10, 0.45), 3)
      )
    )
    data.foreach { point =>
      val metadata = Map[String, Double]()
      val origin: BSV[Double] = new BSV[Double](Array(), Array(), point.closestVertex.size)
      val polytopeProjection = new PolytopeProjection(point.polytope, 10)
      val vertexSolution = polytopeProjection.checkVertexSolution(origin, metadata)
      Assert.assertEquals(truncate(vertexSolution.firstBest), point.closestVertex)
      Assert.assertEquals(truncate(vertexSolution.nextBest), point.nextBestVertex)
      Assert.assertEquals(truncate(polytopeProjection.project(origin, metadata)), point.minima)
    }
  }

  @Test
  def testPolytopeProjection(): Unit = {
    case class PolygonProjectionTest(xHat: BSV[Double], polytope: mutable.Set[BSV[Double]], closestVertex: BSV[Double],
      nextBestVertex: BSV[Double], minima: BSV[Double], finalCorral: Seq[BSV[Double]])
    val data: Seq[PolygonProjectionTest] = Seq(
      // Example from http://essay.utwente.nl/81914/1/Petrov_BA_EEMCS.pdf
      PolygonProjectionTest(
        xHat = new BSV[Double](Array(0, 1), Array(-1, -1), 2),
        polytope = mutable.Set(
          new BSV[Double](Array(0, 1), Array(0, 2), 2),
          new BSV[Double](Array(0, 1), Array(3, 0), 2),
          new BSV[Double](Array(0, 1), Array(-2, 1), 2)),
        closestVertex = new BSV[Double](Array(0, 1), Array(-2, 1), 2),
        nextBestVertex = new BSV[Double](Array(0, 1), Array(3, 0), 2),
        minima = new BSV[Double](Array(0, 1), Array(-0.65, 0.73), 2),
        finalCorral = Seq(
          new BSV[Double](Array(0, 1), Array(-2, 1), 2),
          new BSV[Double](Array(0, 1), Array(-2, 1), 2))
      ),
      // Example from https://arxiv.org/pdf/1710.02608.pdf
      PolygonProjectionTest(
        xHat = new BSV[Double](Array(0, 1, 2), Array(-1, -1, -1), 3),
        polytope = mutable.Set(
          new BSV[Double](Array(0, 1, 2), Array(0.8, 0.9, 0), 3),
          new BSV[Double](Array(0, 1, 2), Array(1.5, -0.5, 0), 3),
          new BSV[Double](Array(0, 1, 2), Array(-1.0, -1.0, 2.0), 3),
          new BSV[Double](Array(0, 1, 2), Array(-4.0, 1.5, 2.0), 3)),
        closestVertex = new BSV[Double](Array(0, 1, 2), Array(1.5, -0.5, 0), 3),
        nextBestVertex = new BSV[Double](Array(0, 1, 2), Array(-1.0, -1.0, 2.0), 3),
        minima = new BSV[Double](Array(0, 1, 2), Array(-0.05, 0.06, 0.56), 3),
        finalCorral = Seq(
          new BSV[Double](Array(0, 1, 2), Array(-4.0, 1.5, 0), 3),
          new BSV[Double](Array(0, 1, 2), Array(1.5, -0.5, 0), 3)
        )
      )
    )
    data.foreach { point =>
      val metadata = Map[String, Double]()
      val polytopeProjection = new PolytopeProjection(point.polytope, 10)
      val vertexSolution = polytopeProjection.checkVertexSolution(point.xHat, metadata)
      Assert.assertEquals(truncate(vertexSolution.firstBest), point.closestVertex)
      Assert.assertEquals(truncate(vertexSolution.nextBest), point.nextBestVertex)
      val solution = polytopeProjection.project(point.xHat, metadata)
      Assert.assertEquals(truncate(solution), point.minima)
      point.finalCorral.foreach{ i =>
        Assert.assertTrue(Math.abs(dot(solution, solution) - dot(solution, i)) > tol)
      }
    }
  }

  @Test
  def testCheckOptimality(): Unit = {
    // We encountered this issue in an application that used boxCutInequality = 1. The error showed up as a
    // singular matrix issue in `affineMinimizer()`, but the bug occurred before that. For this vector, the corral
    // had two vertices {s1, s2} and the best solution x was a convex combination of the two. In the next Wolfe
    // iteration, `BoxCutProjection.candidateVertex()` chose s1 as a candidate vertex to add to the corral. The
    // subsequent call to `checkOptimality(x, s1)` should have returned true, but it did not because of incorrect
    // tolerance value.
    val projection = new BoxCutProjection(maxIter = 10, inequality = true)
    val metadata = Map[String, Double]("boxCut" -> 1)

    val v = new BSV(Array(0, 1, 2), Array(0.6622921563625779, 0.33783878252149596, 0.0), 3)
    val actualProjection = projection.project(v, metadata)
    val expectedProjection = new BSV(Array(0, 1, 2), Array(0.6622266869205409, 0.3377733130794591, 0.0), 3)
    actualProjection should be(expectedProjection)
  }
}

object PolytopeProjectionTest {
  val tol: Double = 1E-02
  case class VertexTest(input: BSV[Double], expected: BSV[Double])

  def truncate(x: BSV[Double], places: Int = 2): BSV[Double] = {
    new BSV(x.index, truncate(x.data, places), x.size)
  }

  def truncate(x: Array[Double], places: Int): Array[Double] = {
    x.map(i => (i * pow(10, places)).round / pow(10, places))
  }
}