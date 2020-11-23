import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassifier, RandomForestClassifier}
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.feature.Stemmer
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DataTypes


object hw1 {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("zybkin")
      .master("local")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    val tokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("words")
      .setPattern("[\\W]")

    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("importantWords")

    val stemmer = new Stemmer()
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("stemmedWords")
      .setLanguage("English")

    val tf = new HashingTF()
      .setNumFeatures(5800)
      .setInputCol(stemmer.getOutputCol)
      .setOutputCol("featureTF")

    val idf = new IDF()
      .setInputCol(tf.getOutputCol)
      .setOutputCol("features")

    val stringIndexer = new StringIndexer()
      .setInputCol("target")
      .setOutputCol("label")

    val gbt = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setMaxIter(32)

    val rfc = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setNumTrees(15)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, stemmer, tf, idf, stringIndexer, gbt))

    val optionsMap = Map(
      "header" -> "true",
      "inferSchema" -> "true",
      "escape" -> "\"",
      "mode" -> "DROPMALFORMED")

    val trainDF = spark.read
      .options(optionsMap)
      .csv("data/train.csv")
      .filter($"target".isNotNull)
      .filter($"text".isNotNull)
      .withColumn("id", $"id".cast(DataTypes.LongType))
      .withColumn("target", $"target".cast(DataTypes.IntegerType))
      .select($"id", $"text", $"target")


    trainDF.show()
    trainDF.printSchema()

    val testDF = spark.read
      .options(optionsMap)
      .csv("data/test.csv")
      .filter($"id".isNotNull)
      .filter($"text".isNotNull)
      .withColumn("id", $"id".cast(DataTypes.LongType))
      .select($"id", $"text")


    testDF.show()
    testDF.printSchema()

    val model = pipeline.fit(trainDF)
    val result = model.transform(testDF)
      .select($"id", $"prediction".as("target").cast(DataTypes.IntegerType))

    val sample = spark.read
      .options(optionsMap)
      .csv("data/sample_submission.csv")
      .withColumn("id", $"id".cast(DataTypes.LongType))
      .select($"id")

    val output = sample.join(result, sample("id").equalTo(result("id")), "left")
      .select(sample("id"), when(result("id").isNull, lit(0)).otherwise($"target").as("target"))

    output.write.options(optionsMap).csv("data/gbt4attempt.csv")

  }
}
