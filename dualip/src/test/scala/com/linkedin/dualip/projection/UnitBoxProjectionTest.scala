package com.linkedin.dualip.projection

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.projection.PolytopeProjectionTest._
import org.testng.Assert
import org.testng.annotations.Test


class UnitBoxProjectionTest {

  @Test
  def testUnitBoxProjection(): Unit = {
    val inputExpectedPairs: Seq[VertexTest] = Seq(
      VertexTest(
        input = new BSV(Array(0, 1, 2), Array(-0.5, 0.8, 1.2), 3),
        expected = new BSV(Array(1, 2), Array(0.8, 1.0), 3)),
      VertexTest(
        input = new BSV(Array(0, 2), Array(2.0, 1.0), 3),
        expected = new BSV(Array(0, 2), Array(1.0, 1.0), 3)),
      VertexTest(
        input = new BSV(Array(0, 1), Array(0.2, 0.8), 2),
        expected = new BSV(Array(0, 1), Array(0.2, 0.8), 2)),
      VertexTest(
        input = new BSV(Array(0, 1, 2, 3), Array(-2.0, 0.0, 0.0, -1.0), 4),
        expected = new BSV(Array(), Array(), 4))
    )
    val unitBoxProjection = new UnitBoxProjection()
    inputExpectedPairs.foreach { item =>
        Assert.assertEquals(unitBoxProjection.project(item.input, Map[String, Double]()), item.expected)
    }
  }
}

