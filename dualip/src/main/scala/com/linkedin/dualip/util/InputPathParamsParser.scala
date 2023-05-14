package com.linkedin.dualip.util

import com.linkedin.dualip.util.DataFormat.DataFormat

/**
 * Case class to represent input path parameters
 *
 * @param ACblocksPath    - Path of matrix A & c encoded as data blocks
 * @param vectorBPath     - Path of vector of budgets b (this should be a dense vector, every itemId should have a constraint)
 * @param format          - The format of input data, e.g. avro or orc
 */
case class InputPaths(ACblocksPath: String, vectorBPath: String, format: DataFormat)

/**
 * Input parameter parser. These are generic input parameters that are shared by most solvers now.
 */
object InputPathParamsParser {
  def parseArgs(args: Array[String]): InputPaths = {
    val parser = new scopt.OptionParser[InputPaths]("Input data parameters parser") {
      override def errorOnUnknownArgument = false

      opt[String]("input.ACblocksPath") required() action { (x, c) => c.copy(ACblocksPath = x) }
      opt[String]("input.vectorBPath") required() action { (x, c) => c.copy(vectorBPath = x) }
      opt[String]("input.format") required() action { (x, c) => c.copy(format = DataFormat.withName(x)) }
    }

    parser.parse(args, InputPaths("", "", DataFormat.AVRO)) match {
      case Some(params) => params
      case _ => throw new IllegalArgumentException(s"Parsing the command line arguments ${args.mkString(", ")} failed")
    }
  }
}