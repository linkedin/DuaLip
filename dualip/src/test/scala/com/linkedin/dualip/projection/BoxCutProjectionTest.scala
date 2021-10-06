/*
 * BSD 2-CLAUSE LICENSE
 *
 * Copyright 2021 LinkedIn Corporation
 * All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

package com.linkedin.dualip.projection

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.blas.VectorOperations.dot
import com.linkedin.dualip.projection.PolytopeProjectionTest._
import org.testng.Assert
import org.testng.annotations.Test


class BoxCutProjectionTest {

  @Test
  def testCandidateVertex(): Unit = {
    case class CheckVertexData(active: Int, input: BSV[Double], farthest: BSV[Double], fScore: Double, closest: BSV[Double], cScore: Double)
    val data: Seq[CheckVertexData] = Seq(
      CheckVertexData(
        active = 2,
        input = new BSV(Array(1, 2, 3), Array(0.8, 0.4, 0.7), 4),
        closest = new BSV(Array(1, 3), Array(1.0, 1.0), 4),
        cScore = 1.5,
        farthest =  new BSV(Array(2, 3), Array(1.0, 1.0), 4),
        fScore = 1.1
      ),
      CheckVertexData(
        active = 2,
        input = new BSV(Array(), Array(), 4),
        closest = new BSV(Array(), Array(), 4),
        cScore = 0.0,
        farthest =  new BSV(Array(), Array(), 4),
        fScore = 0.0
      ),
      CheckVertexData(
        active = 2,
        input = new BSV(Array(1, 2, 3, 5, 7, 8), Array(0.8, 0.5, 0.7, 0.4, 0.9, 0.6), 10),
        closest = new BSV(Array(1, 7), Array(1.0, 1.0), 10),
        cScore = 1.7,
        farthest =  new BSV(Array(2, 5), Array(1.0, 1.0), 10),
        fScore = 0.9
      )
    )
    data.foreach { point =>
      val polytopeProjection = new BoxCutProjection(10)
      val metadata = Map[String, Double]("boxCut" -> point.active)
      val (v1, d1) = polytopeProjection.candidateVertex(point.input, VertexType.Closest, metadata)
      val (v2, d2) = polytopeProjection.candidateVertex(point.input, VertexType.Farthest, metadata)
      Assert.assertEquals(v1, point.closest, "Near check failed")
      Assert.assertTrue(math.abs(d1 - point.cScore) < tol, "Near score failed")
      Assert.assertEquals(v2, point.farthest, "Far check failed")
      Assert.assertTrue(math.abs(d2 - point.fScore) < tol, "Far score failed")
    }
  }

  @Test
  def testCheckVertexSolution(): Unit = {
    case class VertexSolutionTest(active: Int, input: BSV[Double], bestExpected: BSV[Double],
      nextBestExpected: BSV[Double], isOptimalExpected: Boolean, finalCorral: Seq[BSV[Double]])
    val data: Seq[VertexSolutionTest] = Seq(
      VertexSolutionTest(
        active = 2,
        input = new BSV(Array(1, 2, 3, 5, 7, 8), Array(0.8, 0.5, 0.7, 0.4, 0.9, 0.6), 10), // let this be x
        bestExpected = new BSV(Array(1, 7), Array(1.0, 1.0), 10), // y = highest active dimensions of x
        nextBestExpected = new BSV(Array(3, 8), Array(1.0, 1.0), 10), // z = highest active dimensions of (x - y)
        // sorted input in desc. is [0.9, 0.8, 0.7, 0.6, 0.5, 0.4] index corresponding to top 2 elements is [1, 7]
        // (x - y) = [-0.2, 0.5, 0.7, 0.4, -0.1, 0.6] corresponding to the original index
        // sorted (x - y) in desc. is [0.7, 0.6, 0.5, 0.4, -0.1, -0.2] index corresponding to top 2 elements is [3, 8]
        isOptimalExpected = false,
        // (x - y)^T z = 0.7 + 0.6 = 1.3 and (x - y)^T y = -0.1 -0.2 = -0.3 and 1.3 is !< -0.3
        finalCorral = Seq(
          new BSV(Array(1, 2, 3, 5, 7, 8), Array(1.0, 0.0, 0.0, 0.0, 1.0, 0.0), 10),
          new BSV(Array(1, 2, 3, 5, 7, 8), Array(0.0, 0.0, 1.0, 0.0, 0.0, 1.0), 10),
          new BSV(Array(1, 2, 3, 5, 7, 8), Array(0.0, 1.0, 0.0, 1.0, 0.0, 0.0), 10)
        )
      )
    )
    data.foreach { point =>
      val boxedSimplexProjection = new BoxCutProjection(10)
      val metadata = Map[String, Double]("boxCut" -> point.active)
      val vertex = boxedSimplexProjection.checkVertexSolution(point.input, metadata)
      Assert.assertEquals(vertex.firstBest, point.bestExpected)
      Assert.assertEquals(vertex.nextBest, point.nextBestExpected)
      Assert.assertEquals(vertex.isOptimal, point.isOptimalExpected)
      val xStar = boxedSimplexProjection.project(point.input, metadata)
      // The the norm of ||xHat - xStar|| should be equal to
      // (xHat - xStar) dot (xHat - q) for every point q in the final corral
      point.finalCorral.foreach{ i =>
        Assert.assertTrue(Math.abs(dot(xStar, xStar) - dot(xStar, i)) > tol)
      }
    }
  }
}