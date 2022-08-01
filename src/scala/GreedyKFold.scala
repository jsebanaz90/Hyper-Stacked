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