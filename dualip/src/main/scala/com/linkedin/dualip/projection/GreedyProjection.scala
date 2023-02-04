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

import breeze.linalg.{argmax, SparseVector => BSV}


/**
  * Greedy inequality projection. Let j = argmax_i x_i.
  * - If x_j >= 0, return the vector with 1 in the jth position, 0 otherwise.
  * - If x_j < 0, return a zero vector.
  */
class GreedyProjection() extends Projection with Serializable {

  /**
    * Arg max operation to pick the best vertex.
    *
    * @param v    Input vector v.
    * @return     The projected vector.
    */
  override def project(v: BSV[Double], metadata: Metadata): BSV[Double] = {
    val index = argmax(v)
    if (v(index) > 0) {
      new BSV(Array(index), Array(1.0), v.size)
    } else {
      new BSV(Array(), Array(), v.size)
    }
  }

  override def maxNorm(size: Int): Double = ???
  
  override def minNorm(size: Int): Double = ???
}