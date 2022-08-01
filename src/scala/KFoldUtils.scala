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