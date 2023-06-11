from pyspark.sql import SparkSession
from pyspark.sql.functions import count, col
from pyspark.sql.functions import coalesce

spark = SparkSession.builder.appName("Data partition").master("yarn").config("spark.submit.deployMode", "client").getOrCreate()

# Read dataset and join
root_path = "/user/bm106_nyu_edu/1004-project-2023/"
file1 = f"{root_path}interactions_train_small.parquet"
file2 = f"{root_path}tracks_train_small.parquet"
interactions_df = spark.read.parquet(file1)
tracks_df = spark.read.parquet(file2)
data = interactions_df.join(tracks_df, interactions_df.recording_msid == tracks_df.recording_msid).drop(tracks_df.recording_msid)

# If recording_mbid is none, use recording_msid
data = data.withColumn("recording_id", coalesce(data.recording_mbid, data.recording_msid)).drop("recording_mbid", "recording_msid")

# Filter out the inactive users
users = data.groupBy('user_id').agg(count("*").alias("count"))
active_users = users.filter(col("count")>10)
data = active_users.join(data, active_users.user_id == data.user_id, how="left").drop(data.user_id).drop("count")
data.printSchema()

# Partition the data 
train_data, validation_data = data.randomSplit([0.8, 0.2], seed=42)

# Write to parquet files
train_data.write.parquet("/user/wl2841_nyu_edu/final-project-150/train_small.parquet")
validation_data.write.parquet("/user/wl2841_nyu_edu/final-project-150/validation_small.parquet")

spark.stop()
