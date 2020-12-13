import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DataTypes, StructType}


object hw2 {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("zybkin")
      .master("local")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    val scheme = new StructType().add("id", DataTypes.IntegerType).add("text", DataTypes.StringType)

    val recieved = spark.readStream
      .format("socket")
      .option("host", "localhost")
      .option("port", 9999)
      .load()

    val recievedJson =
      recieved.withColumn("json", from_json($"value", scheme))
        .select("json.*")
        .select($"id", $"text")


    val model = PipelineModel.read.load("model/")
    recievedJson.printSchema()
    val result = model.transform(recievedJson.select($"id", $"text"))
      .select($"id", $"prediction".as("target").cast(DataTypes.IntegerType))

    val query = result
      .repartition(1)
      .writeStream
      .outputMode("append")
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("path", "data")
      .option("checkpointLocation", "checkpoint")
      .start()

    query.awaitTermination()
  }
}
