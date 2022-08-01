// Databricks notebook source
// MAGIC %run ./Lib

// COMMAND ----------

spark.sparkContext.getConf.set( "spark.serializer", "org.apache.spark.serializer.KryoSerializer" )
spark.sparkContext.getConf.set( "spark.databricks.io.cache.enabled", "true" )
spark.sparkContext.getConf.getAll

// COMMAND ----------

val hyperStacked = new HyperStacked(
  inputCol = "DEP_DEL15",
  optL1 = true
)

val trainValid = Logger.log(hyperStacked.load(spark,"/mnt/ryan/FLIGHT_train.parquet"), "Training data load time")
val model = Logger.log(hyperStacked.fit(trainValid), "Model fit time")
val test = Logger.log(hyperStacked.load(spark,"/mnt/ryan/FLIGHT_test.parquet"), "Test data load time")
val output = model.transform(test)
val roc = model.evaluate(output, "areaUnderROC")
val pr = model.evaluate(output, "areaUnderPR")
println("ROC Leaderboard")
model.getLeaderBoard(test, "areaUnderROC").foreach(println)
println("PR Leaderboard")
model.getLeaderBoard(test, "areaUnderPR").foreach(println)

// COMMAND ----------

val hyperStacked = new HyperStacked(
  inputCol = 0,
  optL1 = true
)

val trainValid = Logger.log(hyperStacked.load(spark,"/mnt/ryan/SUSY_train.parquet"), "Training data load time")
val model = Logger.log(hyperStacked.fit(trainValid), "Model fit time")
val test = Logger.log(hyperStacked.load(spark,"/mnt/ryan/SUSY_test.parquet"), "Test data load time")
val output = model.transform(test)
val roc = model.evaluate(output, "areaUnderROC")
val pr = model.evaluate(output, "areaUnderPR")
println("ROC Leaderboard")
model.getLeaderBoard(test, "areaUnderROC").foreach(println)
println("PR Leaderboard")
model.getLeaderBoard(test, "areaUnderPR").foreach(println)

// COMMAND ----------

val hyperStacked = new HyperStacked(
  inputCol = "# label",
  optL1 = true
)

val trainValid = Logger.log(hyperStacked.load(spark,"/mnt/ryan/HEPMASS_train.parquet"), "Training data load time")
val model = Logger.log(hyperStacked.fit(trainValid), "Model fit time")
val test = spark.read.parquet("/mnt/ryan/HEPMASS_test.parquet")
val output = model.transform(test)
val roc = model.evaluate(output, "areaUnderROC")
val pr = model.evaluate(output, "areaUnderPR")
println("ROC Leaderboard")
model.getLeaderBoard(test, "areaUnderROC").foreach(println)
println("PR Leaderboard")
model.getLeaderBoard(test, "areaUnderPR").foreach(println)

// COMMAND ----------

val hyperStacked = new HyperStacked(
  inputCol = 0,
  optL1 = true
)

val trainValid = Logger.log(hyperStacked.load(spark,"/mnt/ryan/HIGGS_train.parquet"), "Training data load time")
val model = Logger.log(hyperStacked.fit(trainValid), "Model fit time")
val test = Logger.log(hyperStacked.load(spark,"/mnt/ryan/HIGGS_test.parquet"), "Test data load time")
val output = model.transform(test)
val roc = model.evaluate(output, "areaUnderROC")
val pr = model.evaluate(output, "areaUnderPR")
println("ROC Leaderboard")
model.getLeaderBoard(test, "areaUnderROC").foreach(println)
println("PR Leaderboard")
model.getLeaderBoard(test, "areaUnderPR").foreach(println)
