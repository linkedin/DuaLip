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

