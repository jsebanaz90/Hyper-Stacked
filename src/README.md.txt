This folder contains:

- databricks - the source files exported from databricks ('lib' and 'main' notebooks)
- scala - classes/objects in .scala files extracted from notebook (for readability purposes)

<<USAGE>>

// Initialise HyperStacked
val hyperStacked = new HyperStacked(
  inputCol = "<prediction_column>",
  optL1 = true
)

// Load training data
val trainValid = Logger.log(hyperStacked.load(spark,"/mnt/<blob_storage>/<train_dataset>.parquet"), "Training data load time")

// Train model
val model = Logger.log(hyperStacked.fit(trainValid), "Model fit time")

// Load test data
val test = Logger.log(hyperStacked.load(spark,"/mnt/<blob_storage>/<test_dataset>.parquet"), "Test data load time")

// Predict on test data
val output = model.transform(test)

// Evaluate using chosen metric
val roc = model.evaluate(output, "areaUnderROC")

// Show leader board (optional) 
model.getLeaderBoard(test, "areaUnderROC").foreach(println)