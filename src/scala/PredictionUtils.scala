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