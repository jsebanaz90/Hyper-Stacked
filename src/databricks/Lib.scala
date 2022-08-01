// Databricks notebook source
// DBTITLE 1,Object Logger
/* Object to time and log events. */
object Logger extends Serializable{
  /*
    Method that takes a function, records the time taken for execution and logs to console with an accompanied message
    @Param function   function to be timed
    @Param logMessage message to print to console with time
  */
  def log[A](function: => A, logMessage: String) = {
      val startTime = System.nanoTime
      val output = function
      val timeTaken = (System.nanoTime-startTime)/1e9
      val roundedTime = (math rint timeTaken * 100) / 100
      println(logMessage+" ["+roundedTime+" seconds]")
      output
  }
}

// COMMAND ----------

// DBTITLE 1,Class RandomGridBuilder
import org.apache.spark.ml.param._
import scala.util.Random
import scala.collection.mutable
import scala.math._
/*
  Class to build random grids of parameters. Functionality accounts for different distributions and value types.
  @Param seed an integer seed to be used by a Random object
*/
class RandomGridBuilder(random : Random = new Random(1234)){
  
  /*
    Member to hold added parameter desciptions before being built
  */
  private val paramDescriptors = mutable.Map.empty[Param[_], (String, _, _)]
  
  /*
    Method adds param objects to paramDescriptor member.
    @Param param Param object
    @Param dist  distribution (currently supports "Uniform", "Exponential", "SingleValue")
    @Param min   minimum parameter value
    @Param max   maximum parameter value
  */
  def addGrid[T, U](param: Param[T], dist: String, min: U, max: U) : this.type = {
    paramDescriptors.put(param, (dist, min, max))
    this
  }

  /*
    Method to return a double value in a uniform distribution
    @Param min   minimum parameter value
    @Param max   maximum parameter value
  */
  private def getRandomUniform(min: Double, max: Double) = { 
    (max - min) * random.nextDouble() + min
  }
  
  /*
    Method to return an int value in a uniform distribution
    @Param min   minimum parameter value
    @Param max   maximum parameter value
  */
  private def getRandomUniform(min: Int, max: Int) = {
    random.nextInt(max - min) + min
  }
  
  /*
    Method to return a double value in a exponential distribution
    @Param min   minimum parameter value
    @Param max   maximum parameter value
  */
  private def getRandomExponential(min: Double, max: Double) = {
    val exp = (math.log10(max) - math.log10(min)) * random.nextDouble() + math.log10(min)
    math.pow(10, exp)
  }
  
  /*
    Method to return an int value in a exponential distribution
    @Param min   minimum parameter value
    @Param max   maximum parameter value
  */
  private def getRandomExponential(min: Int, max: Int) = {
    val exp = (math.log10(max) - math.log10(min)) * random.nextDouble() + math.log10(min)
    round(math.pow(10, exp))
  }
  
  /*
    Method to pattern match parameter descriptor tuples and call the respective method
    that will return a random value based on the distribution, min and max specified.
    @Param desciptor tuple containing a distribution, min and max
  */
  private def getRandomValue[T](descriptor : (String, T, T)) = {
    descriptor match {
      case ("Uniform", min: Double, max: Double) =>
        getRandomUniform(min, max)
      case ("Uniform", min: Int, max: Int) =>
        getRandomUniform(min, max)
      case ("Uniform", min: Boolean, max: Boolean) =>
        random.nextBoolean
      case ("Exponential", min: Double, max: Double) =>
        getRandomExponential(min, max)
      case ("Exponential", min: Int, max: Int) =>
        getRandomExponential(min, max)
      case ("SingleValue", value, _) => 
        value
      case (_, _, _) => throw new IllegalArgumentException("Invalid parameter descriptor - see usage for supported types")
    }
  }
  
  /*
    Method that returns an array of parameter configurations of a specified size.
    @Param numModels number of parameter configurations to return
  */
  def build(numModels: Int): Array[ParamMap] = {
    val paramGrid = for { _ <- 0 until numModels } yield {
      val paramMap = new ParamMap()
      paramDescriptors.foreach{ case (param, descriptor) => paramMap.put(param.asInstanceOf[Param[Any]], getRandomValue(descriptor)) }
      paramMap
    }
    paramGrid.toArray
  }
}

// COMMAND ----------

// DBTITLE 1,Object ParameterManager
import org.apache.spark.ml.classification._
import org.apache.spark.ml.param._

/*
  Class to hold the default ranges and distributions of model parameters,
*/
object ParameterManager extends Serializable{

  /*
    Method to return a random grid of parameters for the random forest classifier.
    @Param classifier RandomForestClassifier object
    @Param n          number of configurations to return
  */
  private def getRandomRandomForestParameters(classifier: RandomForestClassifier, n: Int) : Array[ParamMap] = {
    new RandomGridBuilder()
    .addGrid(classifier.maxBins, "SingleValue", 96, None)
    .addGrid(classifier.maxDepth, "Uniform", 5, 20)
    .addGrid(classifier.numTrees, "Uniform", 1, 64)    
    .addGrid(classifier.cacheNodeIds, "SingleValue", true, None) 
    .build(n)
  }
  
  /*
    Method to return a random grid of parameters for the gradient boosted tree classifier.
    @Param classifier GBTClassifer object
    @Param n          number of configurations to return
  */
  private def getRandomGBTParameters(classifier: GBTClassifier, n: Int) : Array[ParamMap] = {
    new RandomGridBuilder()
    .addGrid(classifier.maxBins, "SingleValue", 96, None)
    .addGrid(classifier.maxDepth, "Uniform", 4, 8) 
    .addGrid(classifier.stepSize, "Uniform", 0.01, 0.3)
    .addGrid(classifier.maxIter, "SingleValue", 20, None)  
    .addGrid(classifier.cacheNodeIds, "SingleValue", true, None) 
    .build(n)
  }
  
  /*
    Method to return a random grid of parameters for the linear support vector classifier.
    @Param classifier LinearSVC object
    @Param n          number of configurations to return
  */
  private def getRandomLinearSVCParameters(classifier: LinearSVC, n: Int) : Array[ParamMap] = {
    new RandomGridBuilder()
    .addGrid(classifier.regParam, "Exponential", 1E-3, 1E-1)
    .addGrid(classifier.fitIntercept, "SingleValue", true, None)
    .addGrid(classifier.maxIter, "Uniform", 20, 100)
    .addGrid(classifier.standardization, "SingleValue", true, None)
    .addGrid(classifier.tol, "SingleValue", 1E-6, None)
    .build(n)
  }
    
  /*
    Method to return a random grid of parameters for the logistic regression classifier.
    @Param classifier LogisticRegression object
    @Param n          number of configurations to return
  */
  private def getRandomLogisticRegressionParameters(classifier: LogisticRegression, n: Int) : Array[ParamMap] = {
    new RandomGridBuilder()
    .addGrid(classifier.regParam, "Exponential", 1E-3, 1E-1)
    .addGrid(classifier.elasticNetParam, "Uniform", 0.0, 1.0)
    .addGrid(classifier.maxIter, "Uniform", 20, 100)
    .addGrid(classifier.fitIntercept, "SingleValue", true, None)
    .addGrid(classifier.standardization, "SingleValue", true, None)
    .addGrid(classifier.tol, "SingleValue", 1E-6, None)
    .build(n)
  }
  
  /*
    Method to return a random grid of parameters for the naive bayes classifier.
    @Param classifier NaiveBayes object
    @Param n          number of configurations to return
  */
  private def getRandomNaiveBayesParameters(classifier : NaiveBayes, n: Int) : Array[ParamMap] = {
    new RandomGridBuilder()
    .addGrid(classifier.smoothing, "Uniform", 0.0, 1.0)
    .build(n)
  }
   
  /*
    Method to pattern match a classifier and return the respective random parameter grid
    @Param classifier Classifier object
    @Param n          number of configurations to return
  */
  def getRandomClassifierParameters(classifier: Any, n: Int) : Array[ParamMap] = classifier match {
    case _: RandomForestClassifier         => getRandomRandomForestParameters(classifier.asInstanceOf[RandomForestClassifier], n)
    case _: GBTClassifier                  => getRandomGBTParameters(classifier.asInstanceOf[GBTClassifier], n)
    case _: LinearSVC                      => getRandomLinearSVCParameters(classifier.asInstanceOf[LinearSVC], n)
    case _: LogisticRegression             => getRandomLogisticRegressionParameters(classifier.asInstanceOf[LogisticRegression], n)
    case _: NaiveBayes                     => getRandomNaiveBayesParameters(classifier.asInstanceOf[NaiveBayes], n)
    case _                                 => throw new IllegalArgumentException("Classifier type " + classifier.getClass + " is not supported in ensemble")
  }
  
  /*
    Method to return the default parameters for the random forest classifier
    @Param classifier RandomForestClassifier object
  */
  private def getRandomForestParameters(classifier : RandomForestClassifier) : ParamMap = {
    new ParamMap().put(classifier.cacheNodeIds, true)
  }
  
  /*
    Method to return the default parameters for the gradient boosted tree classifier
    @Param classifier GBTClassifier object
  */
  private def getGBTParameters(classifier : GBTClassifier) : ParamMap = {
    new ParamMap().put(classifier.cacheNodeIds, true).put(classifier.maxIter, 20)
  }
  
  /*
    Method to return the default parameters for the linear support vector classifier
    @Param classifier LinearSVC object
  */
  private def getLinearSVCParameters(classifier : LinearSVC) : ParamMap = {
    new ParamMap().put(classifier.fitIntercept, true).put(classifier.standardization, true).put(classifier.tol, 1E-6).put(classifier.maxIter, 100)
  }
  
  /*
    Method to return the default parameters for the logistic regression classifier
    @Param classifier LogisticRegression object
  */
  private def getLogisticRegressionParameters(classifier : LogisticRegression) : ParamMap = {
    new ParamMap().put(classifier.fitIntercept, true).put(classifier.standardization, true).put(classifier.tol, 1E-6).put(classifier.maxIter, 100)
  }
  
  /*
    Method to return the default parameters for the naive bayes classifier
    @Param classifier NaiveBayes object
  */
  private def getNaiveBayesParameters(classifier : NaiveBayes) : ParamMap = {
    new ParamMap().put(classifier.smoothing, 1.0)
  }
  
  /*
    Method to pattern match a classifier and return the respective default parameter map
    @Param classifier Classifier object
  */
  def getClassifierParameters(classifier: Any) : ParamMap = classifier match {
    case _: RandomForestClassifier         => getRandomForestParameters(classifier.asInstanceOf[RandomForestClassifier])
    case _: GBTClassifier                  => getGBTParameters(classifier.asInstanceOf[GBTClassifier])
    case _: LinearSVC                      => getLinearSVCParameters(classifier.asInstanceOf[LinearSVC])
    case _: LogisticRegression             => getLogisticRegressionParameters(classifier.asInstanceOf[LogisticRegression])
    case _: NaiveBayes                     => getNaiveBayesParameters(classifier.asInstanceOf[NaiveBayes])
    case _                                 => throw new IllegalArgumentException("Classifier type " + classifier.getClass + " is not supported in ensemble")
  }
}

// COMMAND ----------

// DBTITLE 1,Trait StringOrInt
/* Object that can hold a variable of type String or Int */
sealed trait StringOrInt[T]
object StringOrInt {
  implicit val intInstance: StringOrInt[Int] =
    new StringOrInt[Int] {}
  implicit val stringInstance: StringOrInt[String] =
    new StringOrInt[String] {}
}

// COMMAND ----------

// DBTITLE 1,Class DataLoader
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, MinMaxScaler}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.countDistinct
import org.apache.spark.sql.SparkSession


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

// COMMAND ----------

// DBTITLE 1,Object KFoldUtils
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import org.apache.spark.util.SizeEstimator
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.storage.StorageLevel

/* Object that holds utility functions related to k Fold (splitting, sampling, persisting, unpersisting) */
object KFoldUtils extends Serializable{
  
  /*
    Method to return an array of fractions representing split size (e.g [0.25, 0.25, 0.25, 0.25]).
    @Param numFolds  number of folds specified for cross validation
  */
  def getFractionArray(numFolds: Int): Array[Double] = {
    val kFoldFractionArrayBuffer = ArrayBuffer[Double]()
    val fraction = 1.0/numFolds
    for (i <- 0 to numFolds-1){
      kFoldFractionArrayBuffer += fraction
    }
    kFoldFractionArrayBuffer.toArray
  }
  
  /*
    Method to determine if all folds would be cachable.
    @Param dataset   dataset before splitting operation
    @Param numFolds  number of folds specified for cross validation
  */
  def isCachable(dataset: Dataset[Row], numFolds: Int) = {
    val freeCachableMemory = spark.sparkContext.getExecutorMemoryStatus
      .map(kv => kv._2._2)
      .reduce((a, b) => if (a < b) a else b)
    val datasetSize = SizeEstimator.estimate(dataset)

    if (datasetSize*numFolds < freeCachableMemory) true else false
  }
  
  // TODO "Future work" - implement progressive random sampling
  def sample(dataset: Dataset[Row]){
    println("WARN : SizeEstimator suggests that all KFolds will not be cached in memory only. It is recommended that you use a smaller subset.")
  }

  /*
    Method to sample with replacement the training partition of every fold.
    @Param kFolds  array of training, validation tuples for every fold
  */
  def sample(kFolds: Array[(Dataset[Row], Dataset[Row])]) : Array[(Dataset[Row], Dataset[Row])] = {
    kFolds.map{case (train, valid) => (train.sample(true,0.8), valid)}
  }
  
  /*
    Method to get the percentage of the minority class.
    @Param dataset  dataset containing label column
    @Param field    label field name
  */
  def getFieldPercentage(dataset : Dataset[Row], field : String) : Double = {
    val summary = dataset.groupBy(field).count()
    val labelCount : Long = summary.agg(min("count")).first.get(0).asInstanceOf[Long]
    val total : Long = summary.agg(sum("count")).first.get(0).asInstanceOf[Long]
    val percentage = (labelCount.toDouble/total.toDouble)*100
    percentage
  }

  // TODO "Future work" - implement SMOTE
  def balanceDataset(data : Dataset[Row], field : String){ 
    throw new NotImplementedError("Random Sample not balanced - smote not implemented")
  }

  /*
    Method that performs the partitioning into k folds and persists the result.
    @Param numFolds number of folds specified for cross validation
    @Param dataset  dataset to be split into k folds
  */
  def getCachedKFolds(numFolds: Int, dataset: Dataset[Row]) : Array[(Dataset[Row],Dataset[Row])] = {
    dataset.persist(StorageLevel.DISK_ONLY)
    if(!isCachable(dataset, numFolds)){ sample(dataset) }
    val splits = dataset.randomSplit(getFractionArray(numFolds), 1234)
    val cachedKFolds = for (i <- 0 to splits.length-1) yield {
      var train = splits.filter(x => x != splits(i)).reduce(_ union _).cache()
      if(getFieldPercentage(train, "label") < 40){ balanceDataset(train, "label") }
      var valid = splits(i).cache()
      (train,valid)
    }
    dataset.unpersist()
    cachedKFolds.toArray
  }
  
  /*
    Method that performs the partitioning into k folds.
    @Param numFolds number of folds specified for cross validation
    @Param dataset  dataset to be split into k folds
  */
  def getKFolds(numFolds: Int, dataset: Dataset[Row]) : Array[(Dataset[Row],Dataset[Row])] = {
    dataset.cache()
    val splits = dataset.randomSplit(getFractionArray(numFolds), 1234)
    val kFolds = for (i <- 0 to splits.length-1) yield {
      var train = splits.filter(x => x != splits(i)).reduce(_ union _)
      if(getFieldPercentage(train, "label") < 40){ balanceDataset(train, "label") }
      var valid = splits(i)
      (train,valid)
    }
    dataset.unpersist()
    kFolds.toArray
  }
  
  /*
    Method that unpersists all k folds.
    @Param kFolds  array of training/validation tuples for every fold
  */
  def unpersist(kFolds : Array[(Dataset[Row],Dataset[Row])]) : Unit = {
    kFolds.map{ case (train, valid) => 
      train.unpersist() 
      valid.unpersist()
    }
  }
}

// COMMAND ----------

// DBTITLE 1,Object KFold
import org.apache.spark.sql.{DataFrame, Dataset}
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import scala.concurrent.ExecutionContext

/*
  Object that holds functionality to perform k fold cross validation (no hyperparameter optimisation)
*/
object KFold extends Serializable { 
  
  case class Evaluated[F, M <: ClassificationModel[F, M]](val trainedModel: ClassificationModel[F,M], val metric: Double)
  case class ModelMetrics[F, M <: ClassificationModel[F, M]](val parameters: ParamMap, val trainedModels: Array[ClassificationModel[F,M]], var averageMetric: Double, var numEval: Int)
  
  /*
    Method to evaluate a classifier's parameter configuration.
    @Param classifier       Classifier object 
    @Param paramMap         Specifies parameter configuration to be evaluated
    @Param trainValidTuple  Training validation dataset tuple
  */
  private def evaluateParameters[
    F,
    M <: ClassificationModel[F, M],
    E <: Classifier[F, E, M],
  ](
    classifier: Classifier[F,E,M],
    paramMap: ParamMap,
    trainValidTuple: (Dataset[Row],Dataset[Row])
  ) = {
    val evaluator = new BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setRawPredictionCol("rawPrediction")
    .setMetricName("areaUnderROC")
    val model = classifier.fit(trainValidTuple._1, paramMap).asInstanceOf[ClassificationModel[F,M]]
    val metric = evaluator.evaluate(model.transform(trainValidTuple._2, paramMap))
    new Evaluated(model, metric)
  }
  
  /*
    Method to fit and return the results of cross validation.
    @Param classifier  Classifier object to be fit
    @Param kFolds      array of training/validation tuples for every fold     
    @Param maxParams                number of parameter configurations to fit and evaluate
    @Param maxModels                number of models to return
  */
  def fit[
    F,
    M <: ClassificationModel[F, M],
    E <: Classifier[F, E, M],
  ](
    classifier: Classifier[F,E,M],
    kFolds: Array[(Dataset[Row], Dataset[Row])],
    maxParams: Int,
    maxModels: Int
  )(implicit executionContext:ExecutionContext) : Option[Array[(Classifier[F,E,M], ParamMap, Array[ClassificationModel[F,M]], Double)]] =  {
    try{
      if (maxParams <= 0 || maxModels <= 0) return None

      val numFolds = kFolds.length
      val paramMap = ParameterManager.getClassifierParameters(classifier)
      
     val paramMaps = paramMap +: ParameterManager.getRandomClassifierParameters(classifier, maxParams-1)

      val evaluated = kFolds.map{ fold => 
        fold._1.cache(); fold._2.cache() 
        val futureEvaluatedMap = paramMaps.map(
          paramMap => Future[Evaluated[F,M]]{evaluateParameters[F,M,E](classifier, paramMap, fold)}(executionContext)
        )
        val evaluatedMap = futureEvaluatedMap.map(Await.result(_, Duration.Inf))
        fold._1.unpersist(); fold._2.unpersist()
        evaluatedMap
      }
      
      val trainedModels = evaluated.map(fold => fold.map(eval => eval.trainedModel)).transpose
      
      val metrics = evaluated.map(fold => fold.map(eval => eval.metric)).transpose
      val averageMetrics = metrics.map(metricArray => metricArray.reduce(_+_)/numFolds)
      
      val allResults = (paramMaps zip trainedModels zip averageMetrics) map { case ((p,t),a) => (classifier,p,t,a)}
      Some(allResults.sortBy(_._4).reverse.take(maxModels))
      
    } catch {
      case e : Exception => println(classifier.getClass.getSimpleName + " skipped - model failed to train"); None;
    }
  }
  
}

// COMMAND ----------

// DBTITLE 1,Object GreedyKFold
import org.apache.spark.sql.{DataFrame, Dataset}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.ExecutionContext
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.param.ParamMap

/*
  Object that holds functionality to perform greedy k fold cross validation (hyperparameter optimisation)
*/
object GreedyKFold extends Serializable {
  
  case class Evaluated[F, M <: ClassificationModel[F, M]](val trainedModel: ClassificationModel[F,M], val metric: Double)
  case class ModelMetrics[F, M <: ClassificationModel[F, M]](val parameters: ParamMap, val trainedModels: Array[ClassificationModel[F,M]], var averageMetric: Double, var numEval: Int)
  
  /*
    Method to evaluate a classifier's parameter configuration
    @Param classifier       Classifier object 
    @Param paramMap         Specifies parameter configuration to be evaluated
    @Param trainValidTuple  Training validation dataset tuple
  */
  private def evaluateParameters[
    F,
    M <: ClassificationModel[F, M],
    E <: Classifier[F, E, M],
  ](
    classifier: Classifier[F,E,M],
    paramMap: ParamMap,
    trainValidTuple: (Dataset[Row],Dataset[Row])
  ) = {
    val evaluator = new BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setRawPredictionCol("rawPrediction")
    .setMetricName("areaUnderROC")
    val model = classifier.fit(trainValidTuple._1, paramMap).asInstanceOf[ClassificationModel[F,M]]
    val metric = evaluator.evaluate(model.transform(trainValidTuple._2, paramMap))
    new Evaluated(model, metric)
  }

  /*
    Method to return the ModelMetric object containing the best configuration that has not been fully evaluated.
    @Param models   array of ModelMetric objects
    @Param numFolds number of folds specified for cross validation
  */
  private def getBestIncomplete[F, M <: ClassificationModel[F, M]](models: Array[ModelMetrics[F,M]], numFolds: Int) = {
    val sortedIncomplete = models.filter(_.numEval < numFolds).sortBy(_.averageMetric).reverse
    sortedIncomplete(0)
  }

  /*
    Method to return the ModelMetric object containing the best configuration that has been fully evaluated.
    @Param models   array of ModelMetric objects
    @Param numFolds number of folds specified for cross validation
  */
  private def getBestComplete[F, M <: ClassificationModel[F, M]](models: Array[ModelMetrics[F,M]], numFolds: Int) = {
    val sortedComplete = models.filter(_.numEval == numFolds).sortBy(_.averageMetric).reverse
    sortedComplete(0)
  }

  /*
    Method to return all ModelMetric objects containing the configurations that have been fully evaluated.
    @Param models   array of ModelMetric objects
    @Param numFolds number of folds specified for cross validation
  */
  private def getCompleted[F, M <: ClassificationModel[F, M]](models: Array[ModelMetrics[F,M]], numFolds: Int) = {
    models.filter(_.numEval == numFolds)
  }
  
  /*
    Method to fit and return the results of cross validation.
    @Param classifier               Classifier object to be fit
    @Param kFolds                   array of training/validation tuples for every fold     
    @Param maxParams                number of parameter configurations to fit and evaluate
    @Param maxModels                number of models to return
  */
  def fit[
    F,
    M <: ClassificationModel[F, M],
    E <: Classifier[F, E, M],
  ](
    classifier: Classifier[F,E,M],
    kFolds: Array[(Dataset[Row], Dataset[Row])],
    maxParams: Int,
    maxModels: Int
  )(implicit executionContext:ExecutionContext) : Option[Array[(Classifier[F,E,M], ParamMap, Array[ClassificationModel[F,M]], Double)]] =  {

    try {
      if (maxParams <= 0 || maxModels <= 0) return None

      val numFolds = kFolds.length
      val paramMaps = ParameterManager.getRandomClassifierParameters(classifier, maxParams)
      
      kFolds(0)._1.cache(); kFolds(0)._2.cache()
      val firstFoldEvaluatedFutures = paramMaps.map(paramMap => 
        (paramMap, Future[Evaluated[F,M]]{evaluateParameters[F,M,E](classifier, paramMap, kFolds(0))}(executionContext))
      )

      val modelMetrics = firstFoldEvaluatedFutures.map{ case (paramMap, futureEvaluated) => 
        val eval = Await.result(futureEvaluated, Duration.Inf)
        val modelArray = new Array[ClassificationModel[F,M]](numFolds)
        modelArray(0) = eval.trainedModel
        new ModelMetrics(
          parameters = paramMap,
          trainedModels = modelArray,
          averageMetric = eval.metric,
          numEval = 1
      )}
      kFolds(0)._1.unpersist(); kFolds(0)._2.unpersist()

      kFolds.zipWithIndex.map{ case ((train, valid), index) => 
        if(index != 0){
          train.cache() 
          valid.cache()
        }
      }
      
      var n = maxParams
      var max = maxParams*numFolds
      var completedModelCounter = 0 
      while(n < max){
        val model = getBestIncomplete(modelMetrics, numFolds)
        val eval = evaluateParameters(classifier, model.parameters, kFolds(model.numEval))
        model.averageMetric = ((model.averageMetric * model.numEval) + eval.metric)/(model.numEval + 1) 
        model.trainedModels(model.numEval) = eval.trainedModel
        model.numEval += 1
        n += 1
        if (model.numEval == numFolds){
          completedModelCounter += 1
          if(completedModelCounter >= maxModels) n = Int.MaxValue
        }
      }
      KFoldUtils.unpersist(kFolds)

      val completedModels = modelMetrics.filter(_.numEval == numFolds)

      Some(
        completedModels.map(modelMetrics => (
          classifier,
          modelMetrics.parameters,
          modelMetrics.trainedModels,
          modelMetrics.averageMetric
        ))
      )
    } catch {
      case _ : Exception => classifier.getClass.getSimpleName + " skipped - model failed to train"; None;
    }
  }
}

// COMMAND ----------

// DBTITLE 1,Object PredictionUtils
import org.apache.spark.ml.linalg.{Vector, Vectors, DenseVector, SparseVector}

/* 
  Object to get the probability of a binary classification prediction. 
*/
object PredictionUtils{
  
  /*
    Method to pattern match a classification model and return the probability of a prediction value for a feature vector.
    @Param model             ClassificationModel object
    @Param dataPointFeatures Input feature vector 
  */
  def getProbability(
    model: ClassificationModel[Vector, _],
    dataPointFeatures: Vector
  ) : Double = { model match {
    case _: RandomForestClassificationModel => getRandomForestprobability(model.predictRaw(dataPointFeatures)).values(1)
    case _: GBTClassificationModel          => getGradientBoostedprobability(model.predictRaw(dataPointFeatures)).values(1)
    case _: LogisticRegressionModel         => getLogisticRegressionprobability(model.predictRaw(dataPointFeatures)).values(1)
    case _                                  => model.predict(dataPointFeatures)
  }}
  
  /*
    ***Note that this method was taken from the Apache Spark github, added to give access to protected methods.***
  */
  private def getRandomForestprobability(rawPrediction: Vector) = { rawPrediction match {
    case dv: DenseVector =>
      normalizeToProbabilitiesInPlace(dv)
      dv
    case sv: SparseVector => throw new RuntimeException("Unexpected error in RandomForestClassificationModel:" +
          " getRandomForestprobability encountered SparseVector")
  }}
  
  /*
    ***Note that this method was taken from the Apache Spark github, added to give access to protected methods.***
  */
  private def getGradientBoostedprobability(rawPrediction: Vector) = { rawPrediction match {
    case dv: DenseVector =>
      dv.values(0) = 1.0 / (1.0 + math.exp(-2.0 * dv.values(0)))
      dv.values(1) = 1.0 - dv.values(0)
      dv
    case sv: SparseVector => throw new RuntimeException("Unexpected error in GBTClassificationModel:" +
          " getGradientBoostedprobability encountered SparseVector")
  }}
  
  /*
    ***Note that this method was taken from the Apache Spark github, added to give access to protected methods.***
  */
  private def getLogisticRegressionprobability(rawPrediction: Vector) = { rawPrediction match {
    case dv: DenseVector =>
      dv.values(0) = 1.0 / (1.0 + math.exp(-dv.values(0)))
      dv.values(1) = 1.0 - dv.values(0)
      dv
    case sv: SparseVector => throw new RuntimeException("Unexpected error in LogisticRegressionModel:" +
          " getLogisticRegressionprobability encountered SparseVector")
  }}
  
  /*
    ***Note that this method was  taken from the Apache Spark github, added to give access to protected methods.***
  */
  private def normalizeToProbabilitiesInPlace(v: DenseVector): Unit = {
    v.values.foreach(value => require(value >= 0,
      "The input raw predictions should be nonnegative."))
    val sum = v.values.sum
    require(sum > 0, "Can't normalize the 0-vector.")
    var i = 0
    val size = v.size
    while (i < size) {
      v.values(i) /= sum
      i += 1
    }
  }
}

// COMMAND ----------

// DBTITLE 1,Class HyperStackedModel
import org.apache.spark.ml.classification._
import org.apache.spark.ml.linalg.{Vector, Vectors, DenseVector, SparseVector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql._
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.util.ThreadUtils
import scala.concurrent.ExecutionContext
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import org.apache.spark.ml.impl.Utils
import spark.implicits._

case class DataPoint(features: Vector, label: Double)

/*
  Class HyperStackedModel holds trained layers as members and allows the transformation and evaluation of a dataset. 
  Leaderboard and weighting functionality is present to inspect and compare individual model performance. 
  
  @Param _layerOneModels  array of classification models present in the base layer
  @Param _layerTwoModel   classification model present in the meta layer
*/
class HyperStackedModel(_layerOneModels: Array[ClassificationModel[Vector,_]], _layerTwoModel: ClassificationModel[Vector,_]){
  
  /* Accessor */
  val layerOneModels = _layerOneModels
  /* Accessor */
  val layerTwoModel = _layerTwoModel
  /* Member to hold all base layer identifiers */
  val layerOneIdentifiers =  layerOneModels.zipWithIndex.map{ case (model, index) => model.uid + "-" + (index+1)}
 
  /* 
    Method transform passes an input dataset through the respective layers to produce a prediction for every datapoint. 
    @Param input  a dataset that is to be transformed to result in a prediction. This should contain NO training data. 
  */
  def transform(input: Dataset[Row]) = {
    val layerOneOutput = HyperStackedModel.predictLayer(layerOneModels,input)
    val layerTwoOutput = layerTwoModel.transform(layerOneOutput)
    layerTwoOutput
  }
  
  /*
    Method evaluates a model output to give an overall AUC metric.
    @Param modelOutput  a dataset holding a predicted column and true label column, produced by the model after calling 'transform'. 
    @Param metric       metric used for evaluation
  */
  def evaluate(modelOutput: Dataset[Row], metric: String = "areaUnderROC") : Double = {
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName(metric)
    evaluator.evaluate(modelOutput)
  }
  
  /*
    Method evaluates all base layer models and meta model to return a sorted leaderboard based on the AUC metric.
    @Param dataset  a dataset that is to be used to evaluate the models. This should contain NO training data. 
    @Param metric       metric used for evaluation
  */
  def getLeaderBoard(dataset : Dataset[Row], metric: String = "areaUnderROC") = {
    dataset.cache()
    val baseModelsOutput = layerOneModels.map(model => model.transform(dataset))
    val hyperStackedOutput = transform(dataset)
    dataset.unpersist()
    var leaderboard = layerOneIdentifiers.zip(baseModelsOutput.map(model => evaluate(model, metric)))
    leaderboard = leaderboard :+ ("HyperStacked", evaluate(hyperStackedOutput, metric))
    leaderboard.sortBy(_._2).reverse
  }
  
  /*
    Method that returns the weighting given to base models. 
  */
  def getWeights = layerTwoModel match {
      case _ : LogisticRegressionModel => layerOneIdentifiers.zip(layerTwoModel.asInstanceOf[LogisticRegressionModel].coefficients.asInstanceOf[Vector].toArray)
      case _ => throw new IllegalArgumentException("Weighting only available for logistic regression layer two models.")
  }
}

/*
  Companion object to hold static methods.
*/
object HyperStackedModel extends Serializable {
  
  /*
    Method to transform a dataset given an array/layer of models (e.g. base layer). Computes by partition.
    @Param models   array of classification models
    @Param dataset  dataset to be transformed
  */
  private def predictLayer(
    models: Array[ClassificationModel[Vector, _]],
    dataset : Dataset[Row]
  ) = {
    dataset.as[DataPoint].mapPartitions(
      partition => partition.map(
        point => (
          Vectors.dense(models.map(model => PredictionUtils.getProbability(model, point.features))), 
          point.label
        )
      )
    )
    .withColumnRenamed("_1", "features")
    .withColumnRenamed("_2", "label")
  }
}

// COMMAND ----------

// DBTITLE 1,Class HyperStacked
  /*
  Class HyperStacked holds methods to take an input dataset and produce a trained HyperStacked. 
  
  @Param inputCol       target label or integer representing the column number for the target label
  @Param optL1          specifies whether the base layer performs hyperparameter optimisation
  @Param optL2          specifies whether the meta layer performs hyperparameter optimisation
  @Param maxBaseParams  number of base model parameter configurations to fit and evaluate
  @Param maxBaseModels  number of base models to return
  @Param maxMetaParams  number of meta parameter configurations to fit and evaluate
  @Param numFolds       number of cross validation folds
  @Param parallelism    controls how many jobs are sent to the spark scheduler at a time
*/ 
class HyperStacked[T: StringOrInt](
  inputCol: T,
  optL1: Boolean = true,
  optL2: Boolean = false,
  maxBaseParams: Int = 10,
  maxBaseModels: Int = 2,
  maxMetaParams: Int = 1,
  maxMetaModels: Int = 1,
  numFolds: Int = 5,
  parallelism: Int = 10
){
  
  println("HyperStacked initialised : optL1 = " + optL1.toString + ", optL2 = " + optL2.toString + ", parallelism = " + parallelism.toString)
  
  /* Implicit variable to initiate threadpool based on parallelism parameter */
  implicit val executionContext = parallelism match {
    case 1 =>
      ThreadUtils.sameThread
    case n =>
      ExecutionContext.fromExecutorService(ThreadUtils
        .newDaemonCachedThreadPool(s"${this.getClass.getSimpleName}-thread-pool", n))
  }
  
  /*
    Method to load a parquet file based on input path specified.
    @Param  sparkSession  sparkSession object
    @Param  inputPath     absolute file path to parquet file
  */
  def load(spark: SparkSession, inputPath: String): Dataset[Row] = {
    val _inputCol = inputCol
    val loader = new DataLoader(inputPath, _inputCol)
    val dataset = loader.loadParquet(spark)
    val dataset_ = loader.preprocess(dataset)
    dataset_
  }
  
  /*
    Method to split dataset into test and training/validation datasets.
    @Param dataset input dataset that is to be split
    @Param trainValidFraction fraction of data to be used for training/validation
  */
  def split(dataset: Dataset[Row], trainValidFraction: Double) : Array[Dataset[Row]] = {
    if(!(trainValidFraction < 1)) throw new IllegalArgumentException("Invalid trainValidFraction")
    val testFraction = 1.0 - trainValidFraction
    dataset.randomSplit(Array[Double](trainValidFraction, testFraction), 1234)
  }
  
  /*
    Method to train model layers with kfold cross validation and produce a HyperStackedModel.
    @Param dataset training/validation dataset to be used for cross validation.
  */
  def fit(dataset: Dataset[Row]) : HyperStackedModel = {
    val kFolds = Logger.log(KFoldUtils.getKFolds(numFolds, dataset), "K-Fold Split")
    val layerOneCV = Logger.log(HyperStacked.crossValidation(
      kFolds = kFolds, 
      optimise = optL1, 
      maxParams = maxBaseParams, 
      maxModels = maxBaseModels
    ), "Cross validation layer one. Hyperparameter Optimisation : " + optL1.toString())
    println("Base-Learners chosen : " + layerOneCV.classifiers.distinct.map(_.getClass().getSimpleName()).mkString(", "))
    val metaFeatures = Logger.log(HyperStacked.getMetaFeatures(layerOneCV.foldModels, kFolds), "Gathering MetaFeatures")
    
    metaFeatures.write.mode("overwrite").format("parquet").option("compression", "uncompressed").save("/mnt/ryan/tmp.parquet")
    val metaFeatures_ = spark.read.parquet("/mnt/ryan/tmp.parquet")
    
    val metaKFolds = Logger.log(KFoldUtils.getKFolds(numFolds, metaFeatures_), "Meta K-Fold Split")
    val layerTwoCV = Logger.log(HyperStacked.crossValidation(
      kFolds = metaKFolds, 
      optimise = optL2, 
      maxParams = maxMetaParams, 
      maxModels = maxMetaModels
    ),"Cross validation layer two. Hyperparameter Optimisation : " + optL2.toString())
    
    val bestIndex = layerTwoCV.averageMetrics.zipWithIndex.maxBy(_._1)._2
    println("Meta-Learner chosen : " + layerTwoCV.classifiers(bestIndex).getClass().getSimpleName())
    
    val layerOneModels = HyperStacked.trainLayer(layerOneCV.classifiers, layerOneCV.parameters, dataset)
    val layerTwoModel = HyperStacked.trainLayer(layerTwoCV.classifiers(bestIndex), layerTwoCV.parameters(bestIndex), metaFeatures_) 
    new HyperStackedModel(layerOneModels, layerTwoModel)
  }
}

/*
  Companion object to hold static methods.
*/
object HyperStacked extends Serializable {
  
  case class CrossValidationResults(classifiers: Array[Classifier[Vector,_,_]], parameters: Array[ParamMap], foldModels: Array[Array[ClassificationModel[Vector,_]]], averageMetrics: Array[Double])
  
  /*
    Method to perform cross validation. Holds parameter functionality to perform hyper parameter optimimsation, sampling and a varying number of models.
    @Param kFolds                array of train/valid tuples for every k fold
    @Param optimise              boolean parameter specifying whether to perform hyperparameter optimisation 
    @Param maxParams             number of parameter configurations to fit and evaluate
    @Param maxModels             number of models to return
  */
  private def crossValidation (
    kFolds: Array[(Dataset[Row], Dataset[Row])],
    optimise: Boolean,
    maxParams: Int, 
    maxModels: Int
  ) : CrossValidationResults = {
    val crossValidationResults = if (optimise) {
      Array(
        Logger.log(GreedyKFold.fit(new RandomForestClassifier(), kFolds, maxParams, maxModels), "RandomForest"),
        Logger.log(GreedyKFold.fit(new GBTClassifier(), kFolds, maxParams, maxModels), "GradientBoosted"),
        Logger.log(GreedyKFold.fit(new LinearSVC(), kFolds, maxParams, maxModels), "LinearSVC"), 
        Logger.log(GreedyKFold.fit(new LogisticRegression(), kFolds, maxParams, maxModels), "LogisticRegression"), 
        Logger.log(GreedyKFold.fit(new NaiveBayes(), kFolds, maxParams, maxModels), "NaiveBayes") 
      ).filter(!_.isEmpty).map(_.get).flatMap(_.toList)
    }else{
      Array(
        Logger.log(KFold.fit(new RandomForestClassifier(), kFolds, maxParams, maxModels), "RandomForest"),
        Logger.log(KFold.fit(new GBTClassifier(), kFolds, maxParams, maxModels), "GradientBoostedTree"),
        Logger.log(KFold.fit(new LinearSVC(), kFolds, maxParams, maxModels), "LinearSVC"),
        Logger.log(KFold.fit(new LogisticRegression(), kFolds, maxParams, maxModels), "LogisticRegression"),
        Logger.log(KFold.fit(new NaiveBayes(), kFolds, maxParams, maxModels), "NaiveBayes")
      ).filter(!_.isEmpty).map(_.get).flatMap(_.toList)
    }

    require(crossValidationResults.size > 0, "Number of models must be > 0")

    CrossValidationResults(
      classifiers = crossValidationResults.map(result => result._1.asInstanceOf[Classifier[Vector,_,_]]),
      parameters = crossValidationResults.map(result => result._2),
      foldModels = crossValidationResults.map(result => result._3.asInstanceOf[Array[ClassificationModel[Vector,_]]]),
      averageMetrics = crossValidationResults.map(result => result._4)
    )
  }

  /*
    Method to use base layer models to produce a new dataset containing all out of fold predictions of each model.
    @Param layerOneFoldModels array of base layer models for every k fold
    @Param kFolds             array of train/valid tuples for every k fold
  */
  private def getMetaFeatures(
    layerOneFoldModels: Array[Array[ClassificationModel[Vector,_]]],
    kFolds: Array[(Dataset[Row], Dataset[Row])]
  ) = {
    val outOfFoldPredictions = kFolds.zipWithIndex.map { case ((_, valid), numFold) =>
        val foldEnsemble = layerOneFoldModels.map(array => array(numFold))
        valid.as[DataPoint].mapPartitions(
          partition => partition.map(
              point => (
                Vectors.dense(foldEnsemble.map(model => PredictionUtils.getProbability(model, point.features))), 
                point.label
              )
          )
        )(ExpressionEncoder(): Encoder[(Vector,Double)])
      .withColumnRenamed("_1", "features")
      .withColumnRenamed("_2", "label")
    }
    outOfFoldPredictions.reduce(_ union _)
  }
  
  /*
    Method to train layer of models.
    @Param classifiers array of classifiers in layer to be trained
    @Param paramMaps   array of ParamMap objects containing the respective classifier parameters
    @Param dataset     training dataset
  */
  private def trainLayer(
    classifiers: Array[Classifier[Vector,_,_]],
    paramMaps: Array[ParamMap],
    dataset: Dataset[_]
  )(implicit executionContext:ExecutionContext) = {
    val classifierParamTuples = classifiers.zip(paramMaps)
    dataset.cache()
    val modelFutures = classifierParamTuples.map { case (classifier, paramMap) => 
      Future[ClassificationModel[Vector,_]] {
        classifier.fit(dataset, paramMap).asInstanceOf[ClassificationModel[Vector,_]]
      }(executionContext)
    }
    val trainedModels = modelFutures.map(Await.result(_, Duration.Inf))
    dataset.unpersist()
    trainedModels
  }

  /*
    Method to train a single layer model.
    @Param classifier single classifier in layer to be trained 
    @Param paramMap   ParamMap object containing the respective classifier parameters
    @Param dataset    training dataset
  */
  private def trainLayer(
    classifier: Classifier[Vector,_,_],
    paramMap: ParamMap,
    dataset: Dataset[_]
  ) = {
    dataset.cache()
    val trainedModel = classifier.fit(dataset, paramMap).asInstanceOf[ClassificationModel[Vector,_]]
    dataset.unpersist()
    trainedModel
  }
}
