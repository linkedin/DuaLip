package com.linkedin.dualip.solver

import breeze.linalg.SparseVector
import com.linkedin.dualip.problem.MatchingSolverDualObjectiveFunction.toBSV
import com.linkedin.dualip.util.{MapReduceCollectionWrapper, SolverUtility}
import com.linkedin.dualip.util.SolverUtility.SlackMetadata
import com.twitter.algebird.Tuple3Semigroup
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{Dataset, Encoder, SparkSession}
import scala.collection.mutable

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
abstract class MooDistributedRegularizedObjective(b: SparseVector[Double], gamma: Double, enableHighDimOptimization: Boolean = false)
  (implicit spark: SparkSession) extends DualPrimalDifferentiableObjective with Serializable {
  import spark.implicits._

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
    // choose between two implementations of gradient aggregator: they yield similar results but different efficiency
    val (ax, cx, xx) = if(enableHighDimOptimization) {
      twoStepGradientAggregator(partialGradients.value.asInstanceOf[Dataset[PartialPrimalStats]], lambda.size)
    } else {
      oneStepGradientAggregator(partialGradients)
    }

    val axMinusB = toBSV(ax.toArray, b.size) - b // (Ax - b)
    val gradient = axMinusB // gradient is equal to the slack (Ax - b)
    val unregularizedDualObjective = (lambda dot axMinusB) + cx // λ * (Ax -  b) + cx
    val dualObjective = unregularizedDualObjective + xx * gamma / 2.0 // λ * (Ax -  b) + cx + x * x * gamma / 2

    // compute some extra values, i.e. the primal objective and slack metadata
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
   * @param axMinusB
   * @return
   */
  def extraLogging(axMinusB: SparseVector[Double], lambda: SparseVector[Double]): Map[String, String] = {
    Map.empty
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
  def oneStepGradientAggregator(primalStats: MapReduceCollectionWrapper[PartialPrimalStats]): (Array[(Int, Double)], Double, Double) = {
    // component-wise aggregator for partially computed sums of (ax, cx, xx)
    // cx and xx are just doubles.
    // ax is a vector, we represent it in sparse format as Map[Int, Double] because there is
    // a convenient aggregator for Maps.
    val aggregator = new Tuple3Semigroup[Map[Int, Double], Double, Double]
    val (ax, cx, xx) = primalStats.map( { stats =>
      (stats.ax.toMap, stats.cx, stats.xx)
    } , MooDistributedRegularizedObjective.encoder).reduce(aggregator.plus(_, _))
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
    * Method to implement in the child classes, it will be data/problem dependent
    * @param lambda the dual variable
    * @return
    */
  def getPrimalStats(lambda: SparseVector[Double]): MapReduceCollectionWrapper[PartialPrimalStats]
}

object MooDistributedRegularizedObjective {
  /*
   * Create encoder singletons to reuse and prevent re-initialization costs
   */
  val encoder: Encoder[(Map[Int, Double], Double, Double)] = ExpressionEncoder[(Map[Int, Double], Double, Double)]
}
