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