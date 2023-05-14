package com.linkedin.dualip.projection

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.projection.PolytopeProjectionTest.{VertexTest, truncate}
import org.testng.Assert
import org.testng.annotations.Test


class SimplexProjectionTest {

  @Test
  def testSimplexProjection(): Unit = {
    val inputExpectedPairs: Seq[VertexTest] = Seq(
      VertexTest(
        input = new BSV(Array(1, 2), Array(0.8, 0.4), 3),
        expected = new BSV(Array(1, 2), Array(0.7, 0.3), 3)),
      VertexTest(
        input = new BSV(Array(0, 2), Array(2.0, 1.0), 3),
        expected = new BSV(Array(0), Array(1.0), 3)),
      VertexTest(
        input = new BSV(Array(0, 3), Array(1.0, 1.0), 4),
        expected = new BSV(Array(0, 3), Array(0.5, 0.5), 4)),
      VertexTest(
        input = new BSV(Array(), Array(), 4),
        expected = new BSV(Array(), Array(), 4)),
      VertexTest(
        input = new BSV(Array(0, 1), Array(0.0, 0.0), 2),
        expected = new BSV(Array(0, 1), Array(0.5, 0.5), 2)),
      // (0.1, 0, 0.3, 0) should project to (0.25, 0.15, 0.45, 0.15), but we only project the nonzero entries
      VertexTest(
        input = new BSV(Array(0, 2), Array(0.1, 0.3), 4),
        expected = new BSV(Array(0, 2), Array(0.25, 0.45), 4))
    )
    val simplexProjection = new SimplexProjection(checkVertexSolution = true)
    inputExpectedPairs.foreach { item =>
      val answer = simplexProjection.project(item.input, Map[String, Double]())
      Assert.assertEquals(truncate(answer), item.expected)
    }
  }
}

