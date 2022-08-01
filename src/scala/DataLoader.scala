import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, MinMaxScaler}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.countDistinct
import org.apache.spark.sql.SparkSession

/* Object that can hold a variable of type String or Int */
sealed trait StringOrInt[T]
object StringOrInt {
  implicit val intInstance: StringOrInt[Int] =
    new StringOrInt[Int] {}
  implicit val stringInstance: StringOrInt[String] =
    new StringOrInt[String] {}
}

/* Class to load and preprocess remote files ready for transformations */
class DataLoader[T:StringOrInt](inputPath: String, inputLabel: T){

  /* Member that represents if a header exists */
  val hasHeader: Boolean = headerExists(inputLabel)
  /* Member that holds the true column label */
  val label: String = getTrueLabel(inputLabel)
  
  /*
    Method that pattern matches on label type to return the label column's true label
    @Param inputLabel  column name or index
  */
  private def getTrueLabel[T:StringOrInt](inputLabel: T): String = inputLabel match {
    case s : String => s
    case i : Int => "_c" + i.toString() 
  }
  
  /*
    Method that pattern matches on label type to return if a header exists based on inference
    @Param inputLabel  column name or index
  */
  private def headerExists[T:StringOrInt](inputLabel: T): Boolean = inputLabel match {
    case _ : String => true
    case _ : Int => false 
  }
  
  /*
    Method to vectorise and scale feature columns and index the label column.
    @Param dataset  full dataset to be transformed
    @Param features array of feature column names
    @Param label    label column name
  */
  private def transformDataset(dataset: Dataset[Row], features: Array[String], label: String) = {
    val assembler = new VectorAssembler()
      .setInputCols(features)
      .setOutputCol("features")
    val labelIndexer = new StringIndexer()
      .setInputCol(label)
      .setOutputCol("label")

    val transformed = new Pipeline().setStages(Array(assembler,labelIndexer)).fit(dataset).transform(dataset).select("features", "label")
    transformed
  }
  
  /*
    Method to preprocess a dataset ready for transformations
    @Param dataset  full dataset to be transformed
  */
  def preprocess(dataset: Dataset[Row]) : Dataset[Row] = {
    if(!dataset.columns.contains(label))
      throw new IllegalArgumentException("Input label does not exist.")
    if(dataset.select(countDistinct(label)).first.get(0) != 2)
      throw new IllegalArgumentException("Input label is not a binary column")
    val features = dataset.schema.fieldNames.filter(! _.contains(label))
    val dataset_ = transformDataset(dataset, features, label)
    dataset_
  }
  
  /*
    Method to load csv as Spark Dataset
    @Param schema  optional parameter to optimise loading performance
    @Param spark   SparkSession object
  */
  def loadCsv[T](schema: T, spark: SparkSession) : Dataset[Row] = schema match {
    case schema : StructType => spark.read.schema(schema).option("header", hasHeader).csv(inputPath)
    case None => spark.read.option("header", hasHeader).option("inferSchema", "true").csv(inputPath)
  } 
  
  /*
    Method to load parquet file as Spark Dataset 
    @Param spark  SparkSession object
  */
  def loadParquet(spark: SparkSession) = {
    spark.read.parquet(inputPath)
  }
}