package com.linkedin.dualip.objective

import com.linkedin.dualip.util.ProjectionType.ProjectionType
import org.apache.spark.sql.SparkSession

/**
  * This is the loader API that will be implemented by companion objects.
  */
trait DualPrimalObjectiveLoader {
  /**
    * Very generic loader API, we don't know which arguments will be necessary for initialization
    * so we pass command line arguments and let the loader decide what it needs.
    *
    * @param gamma          Currently used by all objectives, @todo think about making gamma a trait.
    * @param projectionType The type of projection used for simple constraints.
    * @param args           Custom args that are parsed by the loader.
    * @param spark          The spark session.
    * @return
    */
  def apply(gamma: Double, projectionType: ProjectionType, args: Array[String])
    (implicit spark: SparkSession): DualPrimalObjective = ???

  /**
    * This apply function is used in situations where the projection type could differ across requests/users (i).
    *
    * @param gamma Value of the quadratic regularizer.
    * @param args  Custom args that are parsed by the loader.
    * @param spark The spark session.
    * @return
    */
  def apply(gamma: Double, args: Array[String])(implicit spark: SparkSession): DualPrimalObjective = ???

  /**
    * This apply function is used in situations where there is no quadratic regularizer (e.g. subgradient methods).
    *
    * @param args  Custom args that are parsed by the loader.
    * @param spark The spark session.
    * @return
    */
  def apply(args: Array[String])(implicit spark: SparkSession): DualPrimalObjective = ???
}