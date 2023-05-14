package com.linkedin.dualip.blas

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.util.VectorOperations.dot
import org.testng.Assert
import org.testng.annotations.Test

class VectorOperationsTest {

  // Utility function to test if two floating point arrays are the same within tolerance
  def isArrayEqual(a: Array[Double], b: Array[Double], tol: Double): Boolean = {
    if (a.length != b.length)
      false

    else {
      for (i <- a.indices) {
        Assert.assertEquals(a(i), b(i), tol)
      }
      true
    }
  }

  @Test
  def testDot(): Unit = {
    val xMap = scala.collection.mutable.Map[Int, BSV[Double]]()
    xMap += (
      0 -> new BSV(Array(0, 1, 3), Array(-1.0, 1.0, 2.0), 4),
      1 -> new BSV(Array(1, 3), Array(1.0, 2.0), 4)
    )

    val yMap = scala.collection.mutable.Map[Int, BSV[Double]]()
    yMap += (
      0 -> new BSV(Array(0, 1, 2), Array(1.0, 2.0, 3.0), 4),
      1 -> new BSV(Array(2, 4), Array(3.0, 4.0), 4)
    )

    val expectedOutputMap = scala.collection.mutable.Map[Int, Double]()
    expectedOutputMap += (
      0 -> 1.0,
      1 -> 0.0
    )

    xMap.foreach(entry =>
      assert(dot(entry._2, yMap(entry._1)).equals(expectedOutputMap(entry._1))))
  }
}