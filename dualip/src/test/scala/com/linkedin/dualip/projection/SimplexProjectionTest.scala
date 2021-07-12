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
import com.linkedin.dualip.projection.PolytopeProjectionTest._
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

