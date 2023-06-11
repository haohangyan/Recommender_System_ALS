from pyspark.sql import SparkSession
from pyspark.sql.functions import col,collect_list,broadcast
from pyspark.sql.types import ArrayType, IntegerType, StructType, StructField 
from pyspark.mllib.evaluation import RankingMetrics
# Create a Spark session
spark = SparkSession.builder.appName("Baseline popularity model with PySpark").master("yarn").config("spark.submit.deployMode", "client").getOrCreate()

# Load the preprocessed data from parquet files
train_interactions_df = spark.read.parquet("/user/hy2664_nyu_edu/final-project-150/train_small.parquet")
test_interactions_df = spark.read.parquet("/user/hy2664_nyu_edu/final-project-150/validation_small.parquet")

temp = train_interactions_df.groupBy('recording_id').count().withColumnRenamed('count', 'listen_count')
temp = temp.sort('listen_count', ascending=False)
popularity_df = temp.join(train_interactions_df, on='recording_id', how='left_outer')
popularity_df.show(5)
popularity_df.printSchema()

for damping_factor in [0, 5000, 10000]:
    popularity_rdd = popularity_df.select("recording_id", "listen_count").rdd.map(lambda row: (row.recording_id, row.listen_count / (row.listen_count + damping_factor)))

    print("popularity finished")#popularity_rdd:  [recording_id, listen_count]

    # Get the set of listened_tracks for each user in the test set
    user_listened_tracks = test_interactions_df.rdd.map(lambda row: (row.user_id, row.recording_id)).groupByKey().mapValues(list)
    #[user,[listened tracks]]

    # Get the top 100 popular tracks
    top_100_popular_tracks = popularity_rdd.sortBy(lambda x: x[1], ascending=False).map(lambda x: x[0]).take(100)

    # Define a function to recommend tracks
    # def recommend_tracks(user_listened_tracks, top_tracks):
    #     return [track for track in top_tracks if track not in user_listened_tracks]

    # Create recommendations for each user
    recommended_rdd = user_listened_tracks.map(lambda x: (x[0], top_100_popular_tracks))
                                                          #recommend_tracks(x[1], top_100_popular_tracks)))

    # Create actual_rdd using test_interactions_df
    #actual_rdd = test_interactions_df.rdd.map(lambda row: (row.user_id, row.recording_id)).groupByKey().mapValues(list)

    # Join actual_rdd with recommended_rdd
    joined_rdd = user_listened_tracks.join(recommended_rdd)

    # Create a RankingMetrics object from the joined_rdd
    metrics = RankingMetrics(joined_rdd.map(lambda x: (x[1][1], x[1][0])))

    map_at_n = metrics.meanAveragePrecisionAt(100)
    ndcg = metrics.ndcgAt(100)
    print(f"MAP@100: {map_at_n}    ndcg: {ndcg}    , damping factor {damping_factor}")

spark.stop()
