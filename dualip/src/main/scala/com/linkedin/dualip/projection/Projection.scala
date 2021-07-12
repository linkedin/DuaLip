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


/*
 *  This file defines major APIs for projecting the primal variable.
 */

/**
  * Generic API of the slate optimizer. It takes the unconstrained variable as the input and projects it to honor the
  * "simple" constraints in the problem definition.
  */
trait Projection {

  type Metadata = Map[String, Double]
  /**
    * @param v - unconstrained primal variable
    * @return projected primal variable
    */
  def project(v: BSV[Double], metadata: Metadata): BSV[Double]

  /**
   * The max norm (`x^Tx`) possible using the defined projection
   * @param size of the sparse vector for the current record
   * @return
   */
  def maxNorm(size: Int): Double = ???

  /**
   * The min norm (`x^Tx`) possible using the defined projection
   * @param size of the sparse vector for the current record
   * @return
   */
  def minNorm(size: Int): Double = ???

  /**
   * To check if the projection returns a vertex solution
   * @param v - projected primal variable
   * @return
   */
  def isVertexSolution(v: BSV[Double], metadata: Metadata): Boolean = ???
}