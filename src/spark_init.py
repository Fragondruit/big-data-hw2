from pyspark.sql import SparkSession

# Initialize a SparkSession
spark = SparkSession.builder \
    .appName("FriendRecommendation") \
    .master("local[*]") \
    .getOrCreate()
sc = spark.sparkContext
