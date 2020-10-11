import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType

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


    val encode = (lat: Double, lng: Double, precision: Int) => {

      val base32 = "0123456789bcdefghjkmnpqrstuvwxyz"

      var (minLat, maxLat) = (-90.0, 90.0)
      var (minLng, maxLng) = (-180.0, 180.0)
      val bits = List(16, 8, 4, 2, 1)

      (0 until precision).map { p => {
        base32 apply (0 until 5).map { i => {
          if (((5 * p) + i) % 2 == 0) {
            val mid = (minLng + maxLng) / 2.0
            if (lng > mid) {
              minLng = mid
              bits(i)
            } else {
              maxLng = mid
              0
            }
          } else {
            val mid = (minLat + maxLat) / 2.0
            if (lat > mid) {
              minLat = mid
              bits(i)
            } else {
              maxLat = mid
              0
            }
          }
        }
        }.reduceLeft((a, b) => a | b)
      }
      }.mkString("")
    }

    val geohash_udf = udf(encode)
    df.select($"price", geohash_udf($"latitude", $"longitude", lit(5)).alias("geohash"))
      .groupBy($"geohash")
      .mean("price")
      .sort($"avg(price)".desc)
      .show(1)

  }
}
