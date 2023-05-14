package com.linkedin.dualip.util

import com.linkedin.dualip.util.DataFormat.DataFormat

/**
 * Case class to represent parallel LP output parameters
 *
 * @param parallelLPOutputPath    - Path of the parallel LP's raw output
 * @param fairnessDualsCount      - The count of fairness duals of each LP
 * @param positionDualsCount      - The count of position duals of each LP
 * @param featuresOutputPath      - Output path of the extracted features from the parallel LP's output
 * @param format                  - The format of output data, e.g. avro or orc
 */
case class ParallelLPOutputParams(parallelLPOutputPath: String, fairnessDualsCount: Int, positionDualsCount: Int,
                                  featuresOutputPath: String, format: DataFormat)

/**
 * Parallel LP Output parameters parser. These are parameters that are used to extract useful features from the parallel
 * LP's output.
 */
object ParallelLPOutputParamsParser {
  def parseArgs(args: Array[String]): ParallelLPOutputParams = {
    val parser = new scopt.OptionParser[ParallelLPOutputParams]("Parallel LP Output parameters parser") {
      override def errorOnUnknownArgument = false

      opt[String]("output.parallelLPOutputPath") required() action { (x, c) => c.copy(parallelLPOutputPath = x) }
      opt[Int]("output.fairnessDualsCount") required() action { (x, c) => c.copy(fairnessDualsCount = x) }
      opt[Int]("output.positionDualsCount") required() action { (x, c) => c.copy(positionDualsCount = x) }
      opt[String]("output.featuresOutputPath") required() action { (x, c) => c.copy(featuresOutputPath = x) }
      opt[String]("output.format") required() action { (x, c) => c.copy(format = DataFormat.withName(x)) }
    }

    parser.parse(args, ParallelLPOutputParams("", 2, 10, "", DataFormat.AVRO)) match {
      case Some(params) => params
      case _ => throw new IllegalArgumentException(s"Parsing the command line arguments ${args.mkString(", ")} failed")
    }
  }
}