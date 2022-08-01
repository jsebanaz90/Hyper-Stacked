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