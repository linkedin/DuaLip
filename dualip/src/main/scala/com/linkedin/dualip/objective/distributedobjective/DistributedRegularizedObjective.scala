package com.linkedin.dualip.objective.distributedobjective

import breeze.linalg.SparseVector
import com.linkedin.dualip.objective.{DualPrimalComputationResult, DualPrimalObjective, PartialPrimalStats}
import com.linkedin.dualip.util.SolverUtility.SlackMetadata
import com.linkedin.dualip.util.VectorOperations.toBSV
import com.linkedin.dualip.util.{ArrayAggregation, SolverUtility}
import com.twitter.algebird.Tuple3Semigroup
import org.apache.log4j.Logger
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.collection.mutable

/**
  * Partial implementation of common objective logic. It takes care of spark
  * aggregations, but getting the primal is left to non-abstract children
  *
  * @param b                         - the constraint vector
  * @param gamma                     - the smoothness parameter
  * @param enableHighDimOptimization - enables optimizations for extremely high dimensional lambda
  *                                  for example, two-stage gradient aggregation
  *                                  cross over point is likely in the range between 100K and 1M, depends on the number of executors
  *                                  use false (default) for smaller dimensional problems
  *                                  use true for higher dimensional problems
  * @param numLambdaPartitions       - used when enableHighDimOptimization=true, dense lambda vectors coming from executors are partitioned
  *                                  for aggregation. The number of partitions should depend on aggregation parallelism and the dimensionality
  *                                  of lambda. A good rule of thumb is to use a multiple of aggregation parallelism to ensure even load
  *                                  but not too high to keep individual partition sizes large (e.g. 1000) for efficiency:
  *                                  numLambdaPartitions \in [5*parallelism, dim(lambda)/1000]
  * @param spark                     - spark session
  */
abstract class DistributedRegularizedObjective(b: SparseVector[Double], gamma: Double,
  enableHighDimOptimization: Boolean = false, numLambdaPartitions: Option[Int] = None)
  (implicit spark: SparkSession) extends DualPrimalObjective with Serializable {

  import DistributedRegularizedObjective._

  override def dualDimensionality: Int = b.size

  /**
    * Implements the main interface method
    *
    * @param lambda              The variable (vector) being optimized
    * @param log                 Key-value pairs used to store logging information for each iteration of the optimizer
    * @param verbosity           Control the logging level
    * @param designInequality    True if Ax <= b, false if Ax = b or have mixed constraints
    * @param mixedDesignPivotNum The pivot number if we have mixed A_1x <= b1 and A_2x = b2, i.e. how many inequality constraints come first
    * @return
    */
  override def calculate(lambda: SparseVector[Double], log: mutable.Map[String, String], verbosity: Int, designInequality: Boolean = true, mixedDesignPivotNum: Int = 0): DualPrimalComputationResult = {
    // compute and aggregate gradients and objective value
    val partialGradients = getPrimalStats(lambda)
    val aggPrimalStats = if (!partialGradients.isEmpty) {
      // choose between two implementations of gradient aggregator: they yield identical results
      // but different efficiency
      if (enableHighDimOptimization) {
        val partitions = numLambdaPartitions.getOrElse(guessNumberOfLambdaPartitions(lambda.size))
        require(partitions <= lambda.size,
          s"Number of lambda aggregation partitions ($partitions) cannot be smaller than dimensionality of lambda ($lambda.size)")
        twoStepGradientAggregator(partialGradients, lambda.size, partitions)
      } else {
        oneStepGradientAggregator(partialGradients)
      }
    }
    else
      PartialPrimalStats(Array[(Int, Double)](), 0.0, 0.0)

    val axMinusB = toBSV(aggPrimalStats.costs, b.size) - b // (Ax - b)
    val gradient = axMinusB // gradient is equal to the slack (Ax - b)
    val unregularizedDualObjective = (lambda dot axMinusB) + aggPrimalStats.objective // λ * (Ax -  b) + cx
    val dualObjective = unregularizedDualObjective + aggPrimalStats.xx * gamma / 2.0 // λ * (Ax -  b) + cx + x * x * gamma / 2

    // compute some extra values
    val primalObjective = aggPrimalStats.objective + aggPrimalStats.xx * gamma / 2.0
    val slackMetadata: SlackMetadata = SolverUtility.getSlack(lambda.toArray, axMinusB.toArray, b.toArray, designInequality, mixedDesignPivotNum)

    // The sum of positive slacks, one of measures of constraints violation useful for logging
    val absoluteConstraintsViolation = axMinusB.toArray.filter(_ > 0.0).sum
    log += ("dual_obj" -> f"$dualObjective%.8e")
    log += ("cx" -> f"${aggPrimalStats.objective}%.8e")
    log += ("feasibility" -> f"${slackMetadata.feasibility}%.6e")
    log += ("λ(Ax-b)" -> f"${lambda dot gradient}%.6e")
    log += ("γ||x||^2/2" -> f"${aggPrimalStats.xx * gamma / 2.0}%.6e")
    log ++= extraLogging(axMinusB, lambda)

    if (verbosity >= 1) {
      log += ("max_pos_slack" -> f"${slackMetadata.maxPosSlack}%.6e")
      log += ("max_zero_slack" -> f"${slackMetadata.maxZeroSlack}%.6e")
      log += ("abs_slack_sum" -> f"$absoluteConstraintsViolation%.6e")
    }
    DualPrimalComputationResult(lambda, dualObjective, unregularizedDualObjective, gradient, primalObjective, axMinusB, slackMetadata)
  }

  /**
    * Method to implement in the child classes, it will be data/problem dependent.
    *
    * @param lambda The dual variable.
    * @return
    */
  def getPrimalStats(lambda: SparseVector[Double]): Dataset[PartialPrimalStats]
}

/**
  * Companion object with gradient aggregation implementation.
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
    * @param primalStats The partial primal statistics.
    * @return
    */
  def oneStepGradientAggregator(primalStats: Dataset[PartialPrimalStats])
    (implicit sparkSession: SparkSession): PartialPrimalStats = {
    import sparkSession.implicits._
    // component-wise aggregator for partially computed sums of (ax, cx, xx)
    // cx and xx are just doubles.
    // ax is a vector, we represent it in sparse format as Map[Int, Double] because there is
    // a convenient aggregator for Maps.
    val aggregator = new Tuple3Semigroup[Map[Int, Double], Double, Double]
    val (ax, cx, xx) = primalStats.map { stats =>
      (stats.costs.toMap, stats.objective, stats.xx)
    }.reduce(aggregator.plus(_, _))
    PartialPrimalStats(ax.toArray, cx, xx)
  }

  /**
    * Does aggregation in the following way:
    * 1. each data partition performs aggregation of the gradients into java Array (dense)
    * 2. we partition array into roughly equal subarrays
    * 3. send subarrays to executors for aggregation, keyed by subarray id.
    * 4. collect aggregated subarrays back to the driver and assemble into single array
    *
    * Example of performance on a real dataset compared to twoStepGradientAggregator:
    * lambdaDim: 1.3M
    * number of lambdas to aggregate: 4000
    * spark.default.parallelism = 200
    * executor memory: 16Gb (not a limiting factor for aggregation)
    *   - Speedup of gradient computation tasks (that also perform initial gradient aggregation).
    *     Median task time improved from 7s to 1s. The speedup of whole stage from 72s to 37s.
    *   - Speedup of gradient aggregation tasks
    *     Median task time improved from 17s to 1s. The speedup of whole stage from 35s to 8s.
    *   - Speedup of the full gradient iteration from 120s to 40-50s. (e.g. >50% improvement).
    *
    * @param primalStats   The partial primal statistics.
    * @param lambdaDim     Dimensionality of the dual variable.
    * @param numPartitions Number of partitions.
    * @param sparkSession  The spark session.
    * @return
    */
  def twoStepGradientAggregator(primalStats: Dataset[PartialPrimalStats], lambdaDim: Int, numPartitions: Int)
    (implicit sparkSession: SparkSession): PartialPrimalStats = {
    val aggregate = primalStats.mapPartitions { partitionIterator =>
      val acxxAgg = new Array[Double](lambdaDim + 2)
      partitionIterator.foreach { stats =>
        val ax = stats.costs
        var i = 0
        while (i < ax.length) {
          val (axIndex, axValue) = ax(i)
          acxxAgg(axIndex) += axValue
          i += 1
        }
        acxxAgg(lambdaDim) += stats.objective
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
    PartialPrimalStats(ax.zipWithIndex.map { case (v, i) => (i, v) }, cx, xx)
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
    * lambdaDim - dimensionality of gradient
    * parallelism - how many "reducers" we use to aggregate gradients from our main tasks
    * numOfGradientTasks - number of partitions of the problem dataset, it determines the number of
    * tasks that compute gradients and hence defines the number of gradient vectors
    * that need aggregation.
    * numOfLambdaPartitions - partitioning of lambda/gradient for efficient aggregation, quantity that
    * this function tries to guess.
    * partitionSize = lambdaDim/numOfLambdaPartitions
    *
    * There are two opposite effects that impact the optimal numOfLambdaPartitions and partitionSize.
    *   - Larger partition size means less overhead in aggregation. Having partition size of ~1000 improved the
    *     speed of aggregation 10X compared to partition of size 1 (one extreme)
    *   - Fewer partitions prevent efficient parallelization of aggregation. I.e. having just 1 partition (another extreme)
    *     would mean all gradient vectors are sent to the same executor effectively DDOS-ing it.
    *
    * We recommend numOfLambdaPartitions to be (4-10)*parallelism, but make sure that the partition size is
    * at least (100-1000). Iif the second condtion is not met - it is a good reason to consider reducing parallelism.
    *
    * The dependency of parameter tuning should be the following
    *  1. DatasetSize + Complexity of projection ==> numOfGradientTasks: make sure data fits into memory and all the projections
    *     are computed fast enough.
    *     2. numOfGradientTasks + lambdaDim ==>  parallelism: Parallelism here is driven by I/O costs, how much data each
    *     aggregation executor can accept fast enough.
    *     3. lambdaDim + parallelism ==> numOfLambdaPartitions
    *
    * @param lambdaDim    Dimensionality of the dual variable.
    * @param sparkSession The spark session.
    * @return Number of partitions.
    */
  def guessNumberOfLambdaPartitions(lambdaDim: Int)(implicit sparkSession: SparkSession): Int = {
    val parallelism = sparkSession.conf.get("spark.default.parallelism").toInt
    // cannot have more partitions than the number of dimensions
    val partitions = math.min(DefaultLambdaPartitionsPerExecutor * parallelism, lambdaDim)
    val partitionSize = lambdaDim / partitions
    if (partitionSize < MinimumRecommendedPartitionSize) logger.warn(s"Lambda partition size ${lambdaDim / partitions} is too small, consider reducing spark.default.parallelism")
    logger.info(s"Gradient dimensionality is $lambdaDim, it is partitioned into $partitions subarrays for aggregation")
    partitions
  }
}
