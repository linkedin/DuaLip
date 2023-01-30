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
import com.linkedin.dualip.util.{ArrayAggregation, SolverUtility}
import com.linkedin.dualip.util.SolverUtility.SlackMetadata
import com.twitter.algebird.Tuple3Semigroup
import org.apache.log4j.Logger
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
case class PartialPrimalStats(ax: Array[(Int, Double)], cx: Double, xx: Double)

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
  * @param numLambdaPartitions       - used when enableHighDimOptimization=true, dense lambda vectors coming from executors are partitioned
  *                                    for aggregation. The number of partitions should depend on aggregation parallelism and the dimensionality
  *                                    of lambda. A good rule of thumb is to use a multiple of aggregation parallelism to ensure even load
  *                                    but not too high to keep individual partition sizes large (e.g. 1000) for efficiency:
  *                                    numLambdaPartitions \in [5*parallelism, dim(lambda)/1000]
  *
  * @param spark - spark session
  */
abstract class DistributedRegularizedObjective(b: SparseVector[Double], gamma: Double,
  enableHighDimOptimization: Boolean = false, numLambdaPartitions: Option[Int] = None)
  (implicit spark: SparkSession) extends DualPrimalDifferentiableObjective with Serializable {
  import DistributedRegularizedObjective._

  override def dualDimensionality: Int = b.size

  /**
    * Implements the main interface method
    * @param lambda               The variable (vector) being optimized
    * @param log                  Key-value pairs used to store logging information for each iteration of the optimizer
    * @param verbosity            Control the logging level
    * @param designInequality     True if Ax <= b, false if Ax = b or have mixed constraints
    * @param mixedDesignPivotNum  The pivot number if we have mixed A_1x <= b1 and A_2x = b2, i.e. how many inequality constraints come first
    * @return
    */
  override def calculate(lambda: SparseVector[Double], log: mutable.Map[String, String], verbosity: Int, designInequality: Boolean = true, mixedDesignPivotNum: Int = 0): DualPrimalDifferentiableComputationResult = {
    // compute and aggregate gradients and objective value
    val partialGradients = getPrimalStats(lambda)
    // choose between two implementations of gradient aggregator: they yield identical results
    // but different efficiency
    val (ax, cx, xx) = if(enableHighDimOptimization) {
      val partitions = numLambdaPartitions.getOrElse(guessNumberOfLambdaPartitions(lambda.size))
      require(partitions <= lambda.size,
        s"Number of lambda aggregation partitions ($partitions) cannot be smaller than dimensionality of lambda ($lambda.size)")
      fasterTwoStepGradientAggregator(partialGradients, lambda.size, partitions)
    } else {
      oneStepGradientAggregator(partialGradients)
    }

    val axMinusB = toBSV(ax, b.size) - b // (Ax - b)
    val gradient = axMinusB // gradient is equal to the slack (Ax - b)
    val unregularizedDualObjective = (lambda dot axMinusB) + cx // λ * (Ax -  b) + cx
    val dualObjective = unregularizedDualObjective + xx * gamma / 2.0 // λ * (Ax -  b) + cx + x * x * gamma / 2

    // compute some extra values
    val primalObjective = cx +  xx * gamma / 2.0
    val slackMetadata: SlackMetadata = SolverUtility.getSlack(lambda.toArray, axMinusB.toArray, b.toArray, designInequality, mixedDesignPivotNum)

    // The sum of positive slacks, one of measures of constraints violation useful for logging
    val absoluteConstraintsViolation =  axMinusB.toArray.filter(_ > 0.0).sum
    log += ("dual_obj" -> f"$dualObjective%.8e")
    log += ("cx" -> f"$cx%.8e")
    log += ("feasibility" -> f"${slackMetadata.feasibility}%.6e")
    log += ("λ(Ax-b)" -> f"${lambda dot gradient}%.6e")
    log += ("γ||x||^2/2" -> f"${xx * gamma / 2.0}%.6e")
    log ++= extraLogging(axMinusB, lambda)

    if (verbosity >= 1) {
      log += ("max_pos_slack" -> f"${slackMetadata.maxPosSlack}%.6e")
      log += ("max_zero_slack" -> f"${slackMetadata.maxZeroSlack}%.6e")
      log += ("abs_slack_sum" -> f"$absoluteConstraintsViolation%.6e")
    }
    DualPrimalDifferentiableComputationResult(lambda, dualObjective, unregularizedDualObjective, gradient, primalObjective, axMinusB, slackMetadata)
  }

  /**
    * To add additional custom logging of Ax-b vector, one may want to log individual important constraints
    * @param axMinusB - result of Ax-b computation
    * @param lambda - current value of dual variable
    * @return - the map to be added to the iteration log (key is column name, value is column value)
    */
  def extraLogging(axMinusB: SparseVector[Double], lambda: SparseVector[Double]): Map[String, String] = {
    Map.empty
  }

  /**
    * Method to implement in the child classes, it will be data/problem dependent
    * @param lambda the dual variable
    * @return
    */
  def getPrimalStats(lambda: SparseVector[Double]): Dataset[PartialPrimalStats]
}

/**
  * Companion object with gradient aggregation implementation
  */
object DistributedRegularizedObjective {
  val logger: Logger = Logger.getLogger(getClass)

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
  def oneStepGradientAggregator(primalStats: Dataset[PartialPrimalStats])
    (implicit sparkSession: SparkSession): (Array[(Int, Double)], Double, Double) = {
    import sparkSession.implicits._
    // component-wise aggregator for partially computed sums of (ax, cx, xx)
    // cx and xx are just doubles.
    // ax is a vector, we represent it in sparse format as Map[Int, Double] because there is
    // a convenient aggregator for Maps.
    val aggregator = new Tuple3Semigroup[Map[Int, Double], Double, Double]
    val (ax, cx, xx) = primalStats.map { stats =>
      (stats.ax.toMap, stats.cx, stats.xx)
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
  @deprecated("Use fasterTwoStepGradientAggregator instead, this slower implementation is kept for reference")
  def twoStepGradientAggregator(primalStats: Dataset[PartialPrimalStats], lambdaDim: Int)
    (implicit sparkSession: SparkSession): (Array[(Int, Double)], Double, Double) = {
    import sparkSession.implicits._
    val cxIndex = -1
    val xxIndex = -2
    val aggregate = primalStats.mapPartitions { partitionIterator =>
      val axAgg = new Array[Double](lambdaDim)
      var cxAgg: Double = 0D
      var xxAgg: Double = 0D
      partitionIterator.foreach { stats =>
        val ax = stats.ax
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
    * Does aggregation in the following way:
    * 1. each data partition performs aggregation of the gradients into java Array (dense)
    * 2. we partition array into roughly equal subarrays
    * 3. send subarrays to executors for aggregation, keyed by subarray id.
    * 4. collect aggregated subarrays back to the driver and assemble into single array
    *
    * Example of performance on a real dataset compared to twoStepGradientAggregator:
    *   lambdaDim: 1.3M
    *   number of lambdas to aggregate: 4000
    *   spark.default.parallelism = 200
    *   executor memory: 16Gb (not a limiting factor for aggregation)
    *   - Speedup of gradient computation tasks (that also perform initial gradient aggregation).
    *     Median task time improved from 7s to 1s. The speedup of whole stage from 72s to 37s.
    *   - Speedup of gradient aggregation tasks
    *     Median task time improved from 17s to 1s. The speedup of whole stage from 35s to 8s.
    *   - Speedup of the full gradient iteration from 120s to 40-50s. (e.g. >50% improvement).
    *
    * @param primalStats
    * @param lambdaDim
    * @param numPartitions
    * @param sparkSession
    * @return
    */
  def fasterTwoStepGradientAggregator(primalStats: Dataset[PartialPrimalStats], lambdaDim: Int, numPartitions: Int)
    (implicit sparkSession: SparkSession): (Array[(Int, Double)], Double, Double) = {
    import sparkSession.implicits._
    val aggregate = primalStats.mapPartitions { partitionIterator =>
      val acxxAgg = new Array[Double](lambdaDim + 2)
      partitionIterator.foreach { stats =>
        val ax = stats.ax
        var i = 0
        while( i < ax.length )
        {
          val (axIndex, axValue) = ax(i)
          acxxAgg(axIndex) += axValue
          i += 1
        }
        acxxAgg(lambdaDim) += stats.cx
        acxxAgg(lambdaDim + 1) += stats.xx
      }
      // partition array
      val x = ArrayAggregation.partitionArray(acxxAgg, numPartitions)
      x.iterator
    }.rdd.reduceByKey(ArrayAggregation.aggregateArrays(_, _)).collect()

    val ax = new Array[Double](lambdaDim)
    var cx = 0.0
    var xx = 0.0
    aggregate.foreach { case (partition, subarray) =>
      val (start, end) = ArrayAggregation.partitionBounds(lambdaDim + 2, numPartitions, partition)
      if (partition == numPartitions - 1) {
        // special case for last partition, as it holds 'xx' and 'cx' in the last two positions
        cx = subarray(subarray.length - 2)
        xx = subarray(subarray.length - 1)
        System.arraycopy(subarray, 0, ax, start, subarray.length - 2)
      } else {
        System.arraycopy(subarray, 0, ax, start, subarray.length)
      }
    }
    (ax.zipWithIndex.map{case (v, i) => (i, v)}, cx, xx)
  }

  /**
    * Default number of lambda partitions per aggregation executor, used in guessNumberOfLambdaPartitions() method.
    * Ideally number of partitions should be 4-10X the number of aggregation executors to guarantee
    * even distribution of records.
    *
    * Users may override guessing logic by setting the total number of partitions manually.
    */
  val DefaultLambdaPartitionsPerExecutor: Int = 4
  val MinimumRecommendedPartitionSize: Int = 100

  /**
    * Guess optimal number of partitions for lambda (gradient) vector used for aggregation.
    * Important quantities to determine the optimal number of partitions:
    *   lambdaDim - dimensionality of gradient
    *   parallelism - how many "reducers" we use to aggregate gradients from our main tasks
    *   numOfGradientTasks - number of partitions of the problem dataset, it determines the number of
    *                        tasks that compute gradients and hence defines the number of gradient vectors
    *                        that need aggregation.
    *   numOfLambdaPartitions - partitioning of lambda/gradient for efficient aggregation, quantity that
    *                           this function tries to guess.
    *   partitionSize = lambdaDim/numOfLambdaPartitions
    *
    * There are two opposite effects that impact the optimal numOfLambdaPartitions and partitionSize.
    *   - Larger partition size means less overhead in aggregation. Having partition size of ~1000 improved the
    *     speed of aggregation 10X compared to partition of size 1 (one extreme)
    *   - Fewer partitions prevent efficient parallelization of aggregation. I.e. having just 1 partition (another extreme)
    *     would mean all gradient vectors are sent to the same executor effectively DDOS-ing it.
    *
    *  We recommend numOfLambdaPartitions to be (4-10)*parallelism, but make sure that the partition size is
    *  at least (100-1000). Iif the second condtion is not met - it is a good reason to consider reducing parallelism.
    *
    *  The dependency of parameter tuning should be the following
    *  1. DatasetSize + Complexity of projection ==> numOfGradientTasks: make sure data fits into memory and all the projections
    *     are computed fast enough.
    *  2. numOfGradientTasks + lambdaDim ==>  parallelism: Parallelism here is driven by I/O costs, how much data each
    *     aggregation executor can accept fast enough.
    *  3. lambdaDim + parallelism ==> numOfLambdaPartitions
    *
    * @param lambdaDim
    * @param sparkSession
    * @return
    */
  def guessNumberOfLambdaPartitions(lambdaDim: Int)(implicit sparkSession: SparkSession): Int = {
    val parallelism = sparkSession.conf.get("spark.default.parallelism").toInt
    // cannot have more partitions than the number of dimensions
    val partitions = math.min(DefaultLambdaPartitionsPerExecutor * parallelism, lambdaDim)
    val partitionSize = lambdaDim/partitions
    if(partitionSize < MinimumRecommendedPartitionSize) logger.warn(s"Lambda partition size ${lambdaDim/partitions} is too small, consider reducing spark.default.parallelism")
    logger.info(s"Gradient dimensionality is $lambdaDim, it is partitioned into $partitions subarrays for aggregation")
    partitions
  }
}
