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