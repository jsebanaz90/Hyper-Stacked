import org.apache.spark.ml.classification._
import org.apache.spark.ml.linalg.{Vector, Vectors, DenseVector, SparseVector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
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