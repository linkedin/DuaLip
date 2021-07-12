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


/**
  * UnitBox projection, 0 <= x_j <= 1
  *
  */
class UnitBoxProjection() extends Projection with Serializable {

  /**
    * Set negative values to 0, set values larger than 1 to 1.
    *
    * @param v    Input vector v
    * @return     The projected vector
    */
  override def project(v: BSV[Double], metadata: Metadata): BSV[Double] = {
    val values = v.data.map(x =>
      if (x > 1) 1.0
      else if (x < 0) 0.0
      else x)

    val result = new BSV(v.index, values, v.size)
    // compact() will have all explicit zeros removed.
    result.compact()
    result
  }

  override def maxNorm(size: Int): Double = 1.0
  override def minNorm(size: Int): Double = 0.0

  override def isVertexSolution(v: BSV[Double], metadata: Metadata): Boolean = {
    v.activeSize == 0 || v.data.forall(x => x == 1.0)
  }
}