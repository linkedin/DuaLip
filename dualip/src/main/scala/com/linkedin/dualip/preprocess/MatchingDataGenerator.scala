package com.linkedin.dualip.preprocess

import com.linkedin.dualip.data.MatchingData
import com.linkedin.dualip.preprocess.CostGenerator.{CONSTANT, CostGenerator, DATA, REWARD}
import com.linkedin.dualip.util.DataFormat.{AVRO, DataFormat}
import com.linkedin.dualip.util.{DataFormat, IOUtility}
import org.apache.log4j.Logger
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.functions.col
import org.apache.spark.sql._

import scala.collection.mutable
import scala.collection.mutable.ListBuffer


/**
  * Generate matching data directly in the DataBlock API. This prevents clients from first splitting the data
  * into two matrices and the solver then trying to join them back
  */
object MatchingDataGenerator {

  val dataFolder: String = "/data" // Encodes the reward (c) and cost (a) information
  val budgetFolder: String = "/budget" // Encodes the budget per item
  val mappingFolder: String = "/mapping" // Mapping from 0 based index to item id
  val metadataFolder: String = "/metadata" // Metadata around the size of the data, global features for re-ranking

  val logger: Logger = Logger.getLogger(getClass)

  case class DataRecord(dataBlockId: Long, itemId: Int, reward: Double, cost: Option[Double] = None)

  case class BudgetRecord(itemId: Int, budget: Double)

  /**
    * Create an index map using the column specifed in a dataframe
    *
    * @param itemData   - the dataframe containing all the items
    * @param itemColumn - column representing itemId
    * @param spark
    * @return
    */
  def getIndexFromDataFrame(itemData: DataFrame, itemColumn: String)(implicit spark: SparkSession): Map[Int, Int] = {
    import spark.implicits._
    itemData
      .select(itemColumn).as[Int]
      .distinct
      .collect
      .zipWithIndex
      .toMap
  }

  /**
    * Create a 0 based index for itemId which helps in creating a sparse vector later on
    *
    * @param params
    * @param spark
    * @return
    */
  def createIndex(params: MatchingDataGeneratorParams)(implicit spark: SparkSession): Map[Int, Int] = {
    val itemData = params.budgetValue match {
      // If budget information is not precomputed we use the data (Ac block data) to figure out the item's index
      case Some(b) => IOUtility.readDataFrame(params.dataBasePath + dataFolder, params.dataFormat)
      // If budget information is available, we use the itemId in the budget to compute the index.
      case None => IOUtility.readDataFrame(params.dataBasePath + budgetFolder, params.dataFormat)
    }
    val itemMapping = getIndexFromDataFrame(itemData, params.constraintDim)
    val saveMapping = spark.createDataFrame(itemMapping.toSeq).toDF("itemId", "index")
    IOUtility.saveDataFrame(saveMapping, params.outputPath + mappingFolder, numPartitions = Option(1))
    itemMapping
  }

  /**
    * Use the index map to generate the final budget data based on the index
    *
    * @param params  - data paths and columns corresponding to the fields
    * @param indexer - index map for itemId
    * @param spark
    */
  def indexBudget(params: MatchingDataGeneratorParams, indexer: Broadcast[Map[Int, Int]])(implicit spark: SparkSession)
  : Dataset[(Int, Double)] = {
    import spark.implicits._
    val budget: Dataset[(Int, Double)] = params.budgetValue match {
      // If we want to hardcode or override the budget information, we do not read the budget path
      case Some(b) => spark.createDataFrame(indexer.value
        .valuesIterator
        .map((_, b))
        .toSeq)
        .toDF("itemId", "budget").as[(Int, Double)]
      // If budget is computed by the user for every itemId, we read the budget from a path
      case None => IOUtility.readDataFrame(params.dataBasePath + budgetFolder, params.dataFormat)
        .select(
          col(params.constraintDim).alias("itemId"),
          col(params.budgetDim).alias("budget"))
        .as[BudgetRecord]
        .flatMap { record =>
          indexer.value.get(record.itemId) match {
            case Some(j) => Some(j, record.budget)
            case None => None
          }
        }
    }

    IOUtility.saveDataFrame(budget.toDF, params.outputPath + budgetFolder, numPartitions = Option(1))
    budget
  }

  /**
    * Use the index map to generate the cost and reward data using the DataBlock API
    *
    * @param params  - data paths and columns corresponding to the fields
    * @param indexer - index map for itemId
    * @param spark
    */
  def indexData(params: MatchingDataGeneratorParams, indexer: Broadcast[Map[Int, Int]])(implicit spark: SparkSession)
  : Dataset[MatchingData] = {
    import spark.implicits._

    val minBlockSize: Int = params.minBlockSize.getOrElse(0)

    val columns: ListBuffer[Column] = ListBuffer(
      col(params.dataBlockDim).alias("dataBlockId"),
      col(params.constraintDim).alias("itemId"),
      col(params.rewardDim).alias("reward")
    )
    if (params.costGenerator == DATA) {
      columns += col(params.costDim.get).alias("cost")
    }

    val data = IOUtility.readDataFrame(params.dataBasePath + dataFolder, params.dataFormat)
      .select(columns: _*)
      .map {
        row: Row =>
          val cost: Option[Double] = params.costGenerator match {
            case DATA => Some(row.getAs[Double]("cost"))
            case _ => None
          }
          DataRecord((row.getAs[Number]("dataBlockId")).longValue(),
            (row.getAs[Number]("itemId")).intValue(),
            row.getAs[Number]("reward").doubleValue(),
            cost)
      }
      .as[DataRecord]
      .groupByKey(_.dataBlockId) // corresponds to the i dimension
      .flatMapGroups { case (blockId, records) =>
        val dat = records.flatMap { record =>
          indexer.value.get(record.itemId) match {
            case Some(j) => {
              // Cost can either be pre-generate by the client, or they specify a flag to use
              // unit cost (assignment) or where cost is equal to the reward.
              val cost: Double = params.costGenerator match {
                case CONSTANT => params.costValue.get
                case REWARD => record.reward
                case DATA => record.cost.get
              }
              Some((j, -record.reward, cost))
            }
            case None => None
          }
        }.toSeq
          .sortBy { case (j, c, a) => -c }

        // Sometimes not every item has a budget, so they never get assgined an id, we drop them
        if (dat.length > minBlockSize)
          Some(MatchingData(blockId.toString, dat, null))
        else
          None
      }

    IOUtility.saveDataFrame(data.toDF, params.outputPath + dataFolder, numPartitions = Option(200))
    data
  }


  /**
    * Entry point to spark job
    *
    * @param args
    */
  def main(args: Array[String]): Unit = {

    implicit val spark = SparkSession
      .builder()
      .appName(getClass.getSimpleName)
      .getOrCreate()

    try {
      val params = MatchingDataGeneratorParamsParser.parseArgs(args)
      // Check if the parameters are specified correctly
      params.costGenerator match {
        case CONSTANT => require(params.costValue.isDefined & params.costDim.isEmpty,
          "\"constant\" cost requires costValue to be set to some value and costDim to be empty")
        case REWARD => require(params.costValue.isEmpty & params.costDim.isEmpty,
          "\"reward\" cost requires both costValue and costDim to be empty")
        case DATA => require(params.costValue.isEmpty & params.costDim.isDefined,
          "\"data\" cost requires costValue to be empty and costDim to be set to some value")
      }
      println(params)

      val indexer: Broadcast[Map[Int, Int]] = spark.sparkContext.broadcast(createIndex(params))
      val data = indexData(params, indexer)
      indexBudget(params, indexer)

      val metadata = mutable.Map[String, String]()
      metadata.put("numRows", indexer.value.size.toString)
      metadata.put("numCols", data.count.toString)
      val saveMetadata = spark.createDataFrame(metadata.toSeq).toDF("key", "value")
      IOUtility.saveDataFrame(saveMetadata, params.outputPath + metadataFolder, numPartitions = Option(1))

    } catch {
      case other: Exception => sys.error("Got an exception: " + other)
    } finally {
      spark.stop()
    }
  }
}

/**
  * Cost can either be pre-generate by the client, or they specify a flag to use:
  *   - unit cost where every item gets a cost of 1.0
  *   - a cost that is equal to the reward, i.e. a_ij = c_ij
  *   - or generate the cost in the input data that will be used as is
  */
object CostGenerator extends Enumeration {
  type CostGenerator = Value
  val CONSTANT = Value("constant")
  val REWARD = Value("reward")
  val DATA = Value("data")
}

/**
  * @param dataFormat    - Use Avro, JSON or ORC data as the input
  * @param dataBasePath  - This dataset contains dataBlockId, itemId, reward and optionally cost information for every record
  * @param dataBlockDim  - Column name corresponding to dataBlockId (i)
  * @param constraintDim - Column name corresponding to itemId (j)
  * @param rewardDim     - Column name corresponding to reward information c_ij
  * @param budgetDim     - Column name corresponding to budget information b_j
  * @param budgetValue   - Override the budget information b_j with the value
  * @param costGenerator - Pre-specified methods to fill in cost when cost is not generated in the data
  * @param costDim       - Column name corresponding to cost information a_ij
  * @param costValue     - Pre-specified cost when costGenerator is set to "constant"
  * @param minBlockSize  - If this is set, we filter out data blocks that have too few items
  * @param outputPath    - Output path to write the processed data
  */
case class MatchingDataGeneratorParams(
  dataFormat: DataFormat = AVRO,
  dataBasePath: String = "",
  dataBlockDim: String = "",
  constraintDim: String = "",
  rewardDim: String = "",
  budgetDim: String = "",
  budgetValue: Option[Double] = None,
  costGenerator: CostGenerator = REWARD,
  costDim: Option[String] = None,
  costValue: Option[Double] = None,
  minBlockSize: Option[Int] = None,
  outputPath: String = ""
)

/**
  * Preprocessing parameters parser
  */
object MatchingDataGeneratorParamsParser {
  def parseArgs(args: Array[String]): MatchingDataGeneratorParams = {
    val parser = new scopt.OptionParser[MatchingDataGeneratorParams]("Parsing matching data generator parameters") {
      override def errorOnUnknownArgument = false

      opt[String]("preprocess.dataFormat") optional() action { (x, c) => c.copy(dataFormat = DataFormat.withName(x)) }
      opt[String]("preprocess.dataBasePath") required() action { (x, c) => c.copy(dataBasePath = x) }
      opt[String]("preprocess.dataBlockDim") required() action { (x, c) => c.copy(dataBlockDim = x) }
      opt[String]("preprocess.constraintDim") required() action { (x, c) => c.copy(constraintDim = x) }
      opt[String]("preprocess.rewardDim") required() action { (x, c) => c.copy(rewardDim = x) }
      opt[String]("preprocess.budgetDim") required() action { (x, c) => c.copy(budgetDim = x) }
      opt[String]("preprocess.budgetValue") optional() action { (x, c) => c.copy(budgetValue = Some(x.toDouble)) }
      opt[String]("preprocess.costGenerator") optional() action { (x, c) => c.copy(costGenerator = CostGenerator.withName(x)) }
      opt[String]("preprocess.costDim") optional() action { (x, c) => c.copy(costDim = Some(x)) }
      opt[String]("preprocess.costValue") optional() action { (x, c) => c.copy(costValue = Some(x.toDouble)) }
      opt[String]("preprocess.minBlockSize") optional() action { (x, c) => c.copy(minBlockSize = Some(x.toInt)) }
      opt[String]("preprocess.outputPath") required() action { (x, c) => c.copy(outputPath = x) }
    }

    parser.parse(args, MatchingDataGeneratorParams()) match {
      case Some(params) => params
      case _ => throw new IllegalArgumentException(s"Parsing the command line arguments ${args.mkString(", ")} failed")
    }
  }
}