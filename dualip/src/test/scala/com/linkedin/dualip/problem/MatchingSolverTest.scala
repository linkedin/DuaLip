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

package com.linkedin.dualip.problem

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.projection.{BoxCutProjection, GreedyProjection, SimplexProjection}
import com.linkedin.dualip.slate.{DataBlock, SingleSlotOptimizer, SlateOptimizer}
import com.linkedin.dualip.solver.AcceleratedGradientDescent
import com.linkedin.spark.common.lib.TestUtils
import org.apache.spark.sql.SparkSession
import org.testng.Assert
import org.testng.annotations.Test

import scala.collection.mutable

// schema for DataFrame in primal test
case class MatchingSolverTestVar(value: Double, items: Array[Int])
case class MatchingSolverTestRow(blockId: String, variables: Array[MatchingSolverTestVar])

class MatchingSolverTest {
  val enableHighDimOptimization = false // unit tests should work with both true/false settings, only using "false" to save runtime

  val a = Map(
    (0, 0) -> 0.307766110869125, (0, 10) -> 0.257672501029447, (0, 20) -> 0.552322433330119, (0, 30) -> 0.0563831503968686,(0, 40) -> 0.468549283919856,
    (1, 1) -> 0.483770735096186, (1, 11) -> 0.812402617651969, (1, 21) -> 0.370320537127554, (1, 31) -> 0.546558595029637, (1, 41) -> 0.170262051047757,
    (2, 2) -> 0.624996477039531, (2, 12) -> 0.882165518123657, (2, 22) -> 0.28035383997485,  (2, 32) -> 0.398487901547924, (2, 42) -> 0.76255108229816,
    (3, 3) -> 0.669021712383255, (3, 13) -> 0.204612161964178, (3, 23) -> 0.357524853432551, (3, 33) -> 0.359475114848465, (3, 43) -> 0.690290528349578,
    (4, 4) -> 0.535811153938994, (4, 14) -> 0.710803845431656, (4, 24) -> 0.538348698290065, (4, 34) -> 0.74897222686559,  (4, 44) -> 0.420101450523362
  )
  val c = Map(
    0 -> -0.307766110869125,   1 -> -0.483770735096186,  2 -> -0.624996477039531,  3 -> -0.669021712383255,  4 -> -0.535811153938994,
    10 -> -0.257672501029447, 11 -> -0.812402617651969, 12 -> -0.882165518123657, 13 -> -0.204612161964178, 14 -> -0.710803845431656,
    20 -> -0.552322433330119, 21 -> -0.370320537127554, 22 -> -0.28035383997485,  23 -> -0.357524853432551, 24 -> -0.538348698290065,
    30 -> -0.0563831503968686,31 -> -0.546558595029637, 32 -> -0.398487901547924, 33 -> -0.359475114848465, 34 -> -0.74897222686559,
    40 -> -0.468549283919856, 41 -> -0.170262051047757, 42 -> -0.76255108229816,  43 -> -0.690290528349578, 44 -> -0.420101450523362
  )
  val metadata: Map[String, Double] = Map[String, Double]("boxCut" -> 2)
  val data: Seq[DataBlock] = (0 to 4).map(i => DataBlock(i.toString, (0 to 4).map(j => j + 10 * i).map(j => (j % 10, c(j), a((j % 10, j)))), metadata))
  val b: Array[Double] = Array(0.7, 0.7, 0.7, 0.7, 0.7)

  // Expected values for this problem were computed with SCS
  val expectedDualObjective: Double = -3.4686
  val expectedLambda: Array[Double] = Array(0.0000000, 0.3327713, 0.3855439, 0.3212216, 0.5130992)

  val expectedPrimalUpperBound: Double = -(0.307766110869125 + 0.204612161964178 + 0.28035383997485 + 0.0563831503968686 + 0.170262051047757)

  @Test
  def testMaxSolver(): Unit = {
    implicit val spark: SparkSession = TestUtils.createSparkSession()
    import spark.implicits._
    spark.sparkContext.setLogLevel("warn")

    val slateOptimizer: SlateOptimizer = new SingleSlotOptimizer(0, new GreedyProjection())
    val f = new MatchingSolverDualObjectiveFunction(spark.createDataset(data), BSV(b), slateOptimizer, 1E-06, enableHighDimOptimization, None)

    val optimizer = new AcceleratedGradientDescent(maxIter = 100)
    val (lambda, value, _) = optimizer.maximize(f, BSV.fill(5)(0.1))
    (0 to 4).foreach { i =>
      Assert.assertTrue(Math.abs(lambda(i) - expectedLambda(i)) < 0.05) // converges closer if we run more iterations
    }
    Assert.assertTrue(Math.abs(value.dualObjective - expectedDualObjective) < 0.05)
  }

  //@Test
  def testPrimal(): Unit = {
    implicit val spark: SparkSession = TestUtils.createSparkSession()
    import spark.implicits._
    spark.sparkContext.setLogLevel("warn")

    val slateOptimizer: SlateOptimizer = new SingleSlotOptimizer(0, new GreedyProjection())
    val f = new MatchingSolverDualObjectiveFunction(spark.createDataset(data), BSV(b), slateOptimizer, 1E-06, enableHighDimOptimization, None)
    // compute value using calculate function
    val value = f.calculate(BSV(expectedLambda), mutable.Map.empty, 1)

    // alternative computation using returned primal solution and hardcoded objective
    val primal = f.getPrimalForSaving(BSV(expectedLambda)).get.as[MatchingSolverTestRow].collect.flatMap { r =>
      r.variables.map { x =>
        require(x.items.length == 1)
        (r.blockId.toInt, x.items(0), x.value)
      }
    }
    val primalObj = primal.map { case (i, j, x) => x * c(i*10 + j) + 1E-6*x*x/2.0 }.sum
    Assert.assertTrue(Math.abs(value.primalObjective - primalObj)<1E-8)
  }

  @Test
  def testSimplexSolver(): Unit = {
    implicit val spark: SparkSession = TestUtils.createSparkSession()
    import spark.implicits._
    spark.sparkContext.setLogLevel("warn")

    val gamma = 1E-3
    val slateOptimizer: SlateOptimizer = new SingleSlotOptimizer(gamma, new SimplexProjection())
    val f = new MatchingSolverDualObjectiveFunction(spark.createDataset(data), BSV(b), slateOptimizer, gamma, enableHighDimOptimization, None)

    val primalUpperBound: Double = expectedPrimalUpperBound + 5 * gamma/2
    Assert.assertTrue(Math.abs(f.getPrimalUpperBound - primalUpperBound) < 0.01)

    val optimizer = new AcceleratedGradientDescent(maxIter = 200)

    val (lambda, value, _) = optimizer.maximize(f, BSV.fill(5)(0.1))
    (0 to 4).foreach { i =>
      Assert.assertTrue(Math.abs(lambda(i) - expectedLambda(i)) < 0.01)
    }
    Assert.assertTrue(Math.abs(value.dualObjective - expectedDualObjective) < 0.01)
  }

  @Test
  def testBoxCutSolver(): Unit = {
    implicit val spark: SparkSession = TestUtils.createSparkSession()
    import spark.implicits._
    spark.sparkContext.setLogLevel("warn")

    val gamma = 1E-6
    val slateOptimizer: SlateOptimizer = new SingleSlotOptimizer(gamma, new BoxCutProjection(1000))
    val f = new MatchingSolverDualObjectiveFunction(spark.createDataset(data), BSV(Array(0.7, 0.7, 0.7, 0.7, 0.7)), slateOptimizer, gamma, enableHighDimOptimization, None)

    val optimizer = new AcceleratedGradientDescent(maxIter = 200)

    val initialLambda = BSV.fill(5)(0.1)
    val (lambda, _, _) = optimizer.maximize(f, initialLambda)
    (0 to 4).foreach { i =>
      Assert.assertTrue(Math.abs(lambda(i) - 1.0) < 0.01)
    }
  }
}

