import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.functions.max
import org.apache.spark.sql.functions.corr
import org.apache.spark.sql.functions.stddev
import org.apache.spark.sql.functions.mean
import org.apache.spark.sql.functions.cos
import org.apache.spark.sql.functions.pow
import org.apache.spark.sql.functions.avg
import org.apache.spark.sql.functions.first
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.functions.udf


object hw0 {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("zybkin")
      .master("local")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    val optionsMap = Map(
      "header" -> "true",
      "inferSchema" -> "true",
      "escape" -> "\"",
      "mode" -> "DROPMALFORMED")

    var df = spark.read
      .options(optionsMap)
      .csv("data/AB_NYC_2019.csv")

    df = df.withColumn("price", df("price").cast(IntegerType))

    println("----------MODE----------")
    df.select($"room_type", $"price")
      .groupBy($"room_type", $"price")
      .count()
      .sort($"count".desc)
      .groupBy($"room_type")
      .agg(first($"price").alias("price"), max($"count").alias("mode"))
      .sort($"mode".desc)
      .show(10)

    println("\n\n----------MEAN----------")
    df.select($"room_type", $"price")
      .groupBy($"room_type")
      .agg(mean($"price").alias("avgPrice"))
      .sort($"avgPrice".desc)
      .show(10)

    println("\n\n----------MEDIAN----------")
    df.createOrReplaceTempView("df")
    spark.sql("select room_type, percentile_approx(price, 0.5) as median from df group by room_type")
      .show()

    println("\n\n----------DISPERSION----------")
    df.select($"room_type", $"price")
      .groupBy($"room_type")
      .agg(stddev($"price").alias("stddevPrice"))
      .select($"room_type", ($"stddevPrice" * $"stddevPrice").alias("dispersion"))
      .show()


    println("\n\n----------THE MOST EXPENSIVE----------")
    df.select($"room_type", $"price")
      .sort($"price".desc)
      .show(1)

    println("\n\n----------THE MOST CHEAPEST----------")
    df.select($"room_type", $"price")
      .sort($"price")
      .show(1)


    println("\n\n----------the Pearson Correlation----------")
    df.select($"price", $"minimum_nights", $"number_of_reviews")
      .agg(
        corr($"price", $"minimum_nights"),
        corr($"price", $"number_of_reviews")
      )
      .show()

    val LATITUDE_5KM = 0.045
    val bound_plus = pow($"longitude".plus(cos($"latitude").multiply(40000).divide(360)), -1).multiply(5)
    val bound_minus = pow($"longitude".minus(cos($"latitude").multiply(40000).divide(360)), -1).multiply(5)
    val cond1 = $"x".between($"latitude", $"latitude".plus(LATITUDE_5KM))
      .and($"y".between($"longitude", bound_plus))
    val cond2 = $"x".between($"latitude", $"latitude".minus(LATITUDE_5KM))
      .and($"y".between($"longitude", bound_plus))

    val cond3 = $"x".between($"latitude", $"latitude".plus(LATITUDE_5KM))
      .and($"y".between($"longitude", bound_minus))
    val cond4 = $"x".between($"latitude", $"latitude".minus(LATITUDE_5KM))
      .and($"y".between($"longitude", bound_minus))


    val data = df.select($"price", $"latitude".alias("x"), $"longitude".alias("y"))
    val data2 = df.select($"id", $"latitude", $"longitude").sort($"latitude", $"longitude")
    data2.join(data, cond1)
      .groupBy($"id", $"latitude", $"longitude")
      .agg(avg($"price").alias("average_price"))
      .sort($"average_price".desc)
      .limit(1)
      .show()

    data2.join(data, cond2)
      .groupBy($"id", $"latitude", $"longitude")
      .agg(avg($"price").alias("average_price"))
      .sort($"average_price".desc)
      .limit(1)
      .show()

    data2.join(data, cond3)
      .groupBy($"id", $"latitude", $"longitude")
      .agg(avg($"price").alias("average_price"))
      .sort($"average_price".desc)
      .limit(1)
      .show()

    data2.join(data, cond4)
      .groupBy($"id", $"latitude", $"longitude")
      .agg(avg($"price").alias("average_price"))
      .sort($"average_price".desc)
      .limit(1)
      .show()
  }
}
