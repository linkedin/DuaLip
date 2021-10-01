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

package com.linkedin.dualip.util

/**
  * The enumeration of supported file format.
  */
object DataFormat extends Enumeration {
  type DataFormat = Value
  val AVRO = Value("avro")
  val ORC = Value("orc")
  val JSON = Value("json")
  val CSV = Value("csv")
}

/**
  * The enumeration of available optimizers
  */
object OptimizerType extends Enumeration {
  type OptimizerType = Value
  val LBFGSB = Value("LBFGSB")
  val LBFGS = Value("LBFGS")
  val AGD = Value("AGD")
  val GD = Value("GD")
}

/**
  * The enumeration of available projections
  */
object ProjectionType extends Enumeration {
  type ProjectionType = Value
  val Greedy = Value("greedy")    // Pick the item with the largest reward for each data block
  val SecondPrice = Value("secondPrice") // Used for greedy allocation in a multi-slot case
  val Simplex = Value("simplex")  // As defined in SimplexProjection, \sum_j x_j = 1
  val SimplexInequality = Value("simplexInequality")  // As defined in SimplexProjection, \sum_j x_j <= 1
  val BoxCut = Value("boxCut")  // As defined in BoxSimplexProjection, \sum_j x_j = k
  val BoxCutInequality = Value("boxCutInequality")  // As defined in BoxSimplexProjection, \sum_j x_j <= k
  val UnitBox = Value("unitBox")  // As defined in UnitBoxProjection, 0 <= x_j <= 1
}

/**
  * The enumeration of return status.
  */
object Status extends Enumeration {
  type Status = Value
  val Running = Value("Running")
  val Converged = Value("Converged")
  val Infeasible = Value("Infeasible")
  val Terminated = Value("Terminated")
  val Failed = Value("Failed")
}