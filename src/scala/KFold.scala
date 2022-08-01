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