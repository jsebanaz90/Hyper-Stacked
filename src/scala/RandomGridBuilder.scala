import org.apache.spark.ml.param._
import scala.util.Random
import scala.collection.mutable
import scala.math._
/*
  Class to build random grids of parameters. Functionality accounts for different distributions and value types.
  @Param seed an integer seed to be used by a Random object
*/
class RandomGridBuilder(random : Random = new Random(1234)){
  
  /*
    Member to hold added parameter desciptions before being built
  */
  private val paramDescriptors = mutable.Map.empty[Param[_], (String, _, _)]
  
  /*
    Method adds param objects to paramDescriptor member.
    @Param param Param object
    @Param dist  distribution (currently supports "Uniform", "Exponential", "SingleValue")
    @Param min   minimum parameter value
    @Param max   maximum parameter value
  */
  def addGrid[T, U](param: Param[T], dist: String, min: U, max: U) : this.type = {
    paramDescriptors.put(param, (dist, min, max))
    this
  }

  /*
    Method to return a double value in a uniform distribution
    @Param min   minimum parameter value
    @Param max   maximum parameter value
  */
  private def getRandomUniform(min: Double, max: Double) = { 
    (max - min) * random.nextDouble() + min
  }
  
  /*
    Method to return an int value in a uniform distribution
    @Param min   minimum parameter value
    @Param max   maximum parameter value
  */
  private def getRandomUniform(min: Int, max: Int) = {
    random.nextInt(max - min) + min
  }
  
  /*
    Method to return a double value in a exponential distribution
    @Param min   minimum parameter value
    @Param max   maximum parameter value
  */
  private def getRandomExponential(min: Double, max: Double) = {
    val exp = (math.log10(max) - math.log10(min)) * random.nextDouble() + math.log10(min)
    math.pow(10, exp)
  }
  
  /*
    Method to return an int value in a exponential distribution
    @Param min   minimum parameter value
    @Param max   maximum parameter value
  */
  private def getRandomExponential(min: Int, max: Int) = {
    val exp = (math.log10(max) - math.log10(min)) * random.nextDouble() + math.log10(min)
    round(math.pow(10, exp))
  }
  
  /*
    Method to pattern match parameter descriptor tuples and call the respective method
    that will return a random value based on the distribution, min and max specified.
    @Param desciptor tuple containing a distribution, min and max
  */
  private def getRandomValue[T](descriptor : (String, T, T)) = {
    descriptor match {
      case ("Uniform", min: Double, max: Double) =>
        getRandomUniform(min, max)
      case ("Uniform", min: Int, max: Int) =>
        getRandomUniform(min, max)
      case ("Uniform", min: Boolean, max: Boolean) =>
        random.nextBoolean
      case ("Exponential", min: Double, max: Double) =>
        getRandomExponential(min, max)
      case ("Exponential", min: Int, max: Int) =>
        getRandomExponential(min, max)
      case ("SingleValue", value, _) => 
        value
      case (_, _, _) => throw new IllegalArgumentException("Invalid parameter descriptor - see usage for supported types")
    }
  }
  
  /*
    Method that returns an array of parameter configurations of a specified size.
    @Param numModels number of parameter configurations to return
  */
  def build(numModels: Int): Array[ParamMap] = {
    val paramGrid = for { _ <- 0 until numModels } yield {
      val paramMap = new ParamMap()
      paramDescriptors.foreach{ case (param, descriptor) => paramMap.put(param.asInstanceOf[Param[Any]], getRandomValue(descriptor)) }
      paramMap
    }
    paramGrid.toArray
  }
}