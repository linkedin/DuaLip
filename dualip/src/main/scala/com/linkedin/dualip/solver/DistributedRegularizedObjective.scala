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
 
package com.linkedin.dualip.solver

import breeze.linalg.SparseVector
import com.linkedin.dualip.blas.VectorOperations.toBSV
import com.linkedin.dualip.util.SolverUtility
import com.linkedin.dualip.util.SolverUtility.SlackMetadata
import com.twitter.algebird.Tuple3Semigroup
import org.apache.spark.sql.{Dataset, SparkSession}
import scala.collection.mutable

/**
  * Data encapsulates sufficient statistics of partial (distributed) primal solution,
  * necessary for gradient computation.
  * Note, the actual primal is not returned because it is not necessary for optimization
  *       and its format may be problem-specific (e.g. slate optimization case).
  * @param ax - Ax vector (contribution of this segment into constraints), in sparse format
  * @param cx - contribution to objective (without the regularization term)
  * @param xx - (x dot x) to compute solution norm
  */
case class PartialPrimalStats(ax: Map[Int, Double], cx: Double, xx: Double)

/**
  * Partial implementation of common objective logic. It takes care of spark
  * aggregations, but getting the primal is left to non-abstract children
  * @param b - the constraint vector
  * @param gamma - the smoothness parameter
  * @param enableHighDimOptimization - enables optimizations for extremely high dimensional lambda
  *                                    for example, two-stage gradient aggregation
  *                                    cross over point is likely in the range between 100K and 1M, depends on the number of executors
  *                                    use false (default) for smaller dimensional problems
  *                                    use true for higher dimensional problems
  *
  * @param spark - spark session
  */
abstract class DistributedRegularizedObjective(b: SparseVector[Double], gamma: Double, enableHighDimOptimization: Boolean = false)
  (implicit spark: SparkSession) extends DualPrimalDifferentiableObjective with Serializable {
  import spark.implicits._

  override def dualDimensionality: Int = b.size

  /**
    * Implements the main interface method
    * @param lambda   The variable (vector) being optimized
    * @param log      Key-value pairs used to store logging information for each iteration of the optimizer
    * @param verbosity  Control the logging level
    * @return
    */
  override def calculate(lambda: SparseVector[Double], log: mutable.Map[String, String], verbosity: Int): DualPrimalDifferentiableComputationResult = {
    // compute and aggregate gradients and objective value
    val partialGradients = getPrimalStats(lambda)
    // choose between two implementations of gradient aggregator: they yield similar results
    // but different efficiency
    val (ax, cx, xx) = if(enableHighDimOptimization) {
      twoStepGradientAggregator(partialGradients, lambda.size)
    } else {
      oneStepGradientAggregator(partialGradients)
    }

    val axMinusB = toBSV(ax.toArray, b.size) - b // (Ax - b)
    val gradient = axMinusB // gradient is equal to the slack (Ax - b)
    val unregularizedDualObjective = (lambda dot axMinusB) + cx // λ * (Ax -  b) + cx
    val dualObjective = unregularizedDualObjective + xx * gamma / 2.0 // λ * (Ax -  b) + cx + x * x * gamma / 2

    // compute some extra values
    val primalObjective = cx +  xx * gamma / 2.0
    val slackMetadata: SlackMetadata = SolverUtility.getSlack(lambda.toArray, axMinusB.toArray, b.toArray)

    // The sum of positive slacks, one of measures of constraints violation useful for logging
    val absoluteConstraintsViolation =  axMinusB.toArray.filter(_ > 0.0).sum
    log += ("dual_obj" -> f"$dualObjective%.8e")
    log += ("cx" -> f"$cx%.8e")
    log += ("feasibility" -> f"${slackMetadata.feasibility}%.6e")
    log += ("λ(Ax-b)" -> f"${lambda dot gradient}%.6e")
    log += ("γ||x||^2/2" -> f"${xx * gamma / 2.0}%.6e")

    if (verbosity >= 1) {
      log += ("max_pos_slack" -> f"${slackMetadata.maxPosSlack}%.6e")
      log += ("max_zero_slack" -> f"${slackMetadata.maxZeroSlack}%.6e")
      log += ("abs_slack_sum" -> f"$absoluteConstraintsViolation%.6e")
    }
    DualPrimalDifferentiableComputationResult(lambda, dualObjective, unregularizedDualObjective, gradient, primalObjective, axMinusB, slackMetadata)
  }

  /**
    * Function to aggregate gradients from executors and send them to the driver.
    * Performs one step aggregation where results from executors are send directly to the driver.
    * This approach does not scale well with very high dimensional problems, the driver cannot handle
    * additions of too many high dimensional vectors.
    *
    * For 100K dimensional lambda and hundreds of executors this approach seems to perform well and the
    * bottleneck of computation is on executors. For 1M dimensional lambda this approach no longer works.
    *
    * @param primalStats the partial primal statistics
    * @return
    */
  def oneStepGradientAggregator(primalStats: Dataset[PartialPrimalStats]): (Array[(Int, Double)], Double, Double) = {
    // component-wise aggregator for partially computed sums of (ax, cx, xx)
    // cx and xx are just doubles.
    // ax is a vector, we represent it in sparse format as Map[Int, Double] because there is
    // a convenient aggregator for Maps.
    val aggregator = new Tuple3Semigroup[Map[Int, Double], Double, Double]
    val (ax, cx,xx) = primalStats.map { stats =>
      (stats.ax, stats.cx, stats.xx)
    }.reduce(aggregator.plus(_, _))
    (ax.toArray, cx, xx)
  }

  /**
    * Aggregates gradients from executors. Performs operation in two steps:
    * gradients are computed on executors, partitioned by "j" (vector dimension index) and
    * sent to multiple reducers for aggregation and then the results are collected on driver.
    *
    * Works better than oneStepGradientAggregator for very high dimensional lambdas (i.e. 1M)
    * @param primalStats the partial primal statistics
    * @return
    */
  def twoStepGradientAggregator(primalStats: Dataset[PartialPrimalStats], lambdaDim: Int): (Array[(Int, Double)], Double, Double) = {
    val cxIndex = -1
    val xxIndex = -2
    val aggregate = primalStats.mapPartitions { partitionIterator =>
      val axAgg = new Array[Double](lambdaDim)
      var cxAgg: Double = 0D
      var xxAgg: Double = 0D
      partitionIterator.foreach { stats =>
        val ax = stats.ax.toArray
        var i = 0
        while( i < ax.length )
        {
          val (axIndex, axValue) = ax(i)
          axAgg(axIndex) += axValue
          i += 1
        }
        cxAgg += stats.cx
        xxAgg += stats.xx
      }
      // prepare a key-value list for regular spark key-based aggregation
      // note that cx, xx are given unique keys to bundle aggregation
      // this is a hack, but it works faster than splitting aggregation into two steps
      (Seq((cxIndex, cxAgg), (xxIndex, xxAgg)) ++: axAgg.toSeq.zipWithIndex.map { case (l,r) => (r,l) }).toIterator
    }.rdd.reduceByKey(_ + _).collect()

    // separate ax, cx and xx from flat data structure
    val (ax, cxAndxx) = aggregate.partition { case (index, _) => index != cxIndex && index != xxIndex }
    val cx = cxAndxx.find { _._1 == cxIndex }.get._2
    val xx = cxAndxx.find { _._1 == xxIndex }.get._2
    (ax, cx, xx)
  }

  /**
    * Method to implement in the child classes, it will be data/problem dependent
    * @param lambda the dual variable
    * @return
    */
  def getPrimalStats(lambda: SparseVector[Double]): Dataset[PartialPrimalStats]
}
