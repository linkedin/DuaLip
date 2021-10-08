package com.linkedin.dualip.util

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