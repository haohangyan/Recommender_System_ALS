from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics

spark = SparkSession.builder.appName("Baseline popularity model with PySpark").master("yarn").config("spark.submit.deployMode", "client").getOrCreate()

# Load the preprocessed data from parquet files
train_interactions_df = spark.read.parquet("/user/wl2841_nyu_edu/final-project-150/train_small.parquet")
test_interactions_df = spark.read.parquet("/user/wl2841_nyu_edu/final-project-150/validation_small.parquet")

# Calculate the popularity for each recording in the training set
popularity_rdd = train_interactions_df.groupBy('recording_id').count().rdd.map(lambda row: (row.recording_id, row['count']))

# Apply a damping factor to the popularity values
damping_factor = [0, 100, 1000, 10000]
for beta in damping_factor:
    popularity_rdd = popularity_rdd.map(lambda x: (x[0], x[1] / (x[1] + beta)))

    # Get the top 100 popular tracks
    top_100_popular_tracks = popularity_rdd.sortBy(lambda x: x[1], ascending=False).map(lambda x: x[0]).take(100)

    # Create an RDD with user_id as the key and a list of listened tracks as the value for the test set
    user_listened_tracks = test_interactions_df.rdd.map(lambda row: (row.user_id, row.recording_id)).groupByKey().mapValues(list)
    user_listened_tracks_train = train_interactions_df.rdd.map(lambda row: (row.user_id, row.recording_id)).groupByKey().mapValues(list)
    
    # Recommend the top 100 popular tracks for each user
    recommended_rdd = user_listened_tracks.map(lambda x: (x[0], top_100_popular_tracks))
    recommended_rdd = user_listened_tracks_train.map(lambda x: (x[0], top_100_popular_tracks))
    
    # Join actual_rdd with recommended_rdd
    joined_rdd = user_listened_tracks.join(recommended_rdd)
    joined_rdd_train = user_listened_tracks_train.join(recommended_rdd)
    
    # Create a RankingMetrics object from the joined_rdd
    metrics = RankingMetrics(joined_rdd.map(lambda x: (x[1][1], x[1][0])))
    metrics_train = RankingMetrics(joined_rdd_train.map(lambda x: (x[1][1], x[1][0])))

    ap = metrics.meanAveragePrecisionAt(100)
    ap_train = metrics_train.meanAveragePrecisionAt(100)
    
    ndcg = metrics.ndcgAt(100)
    ndcg_train = metrics_train.ndcgAt(100)
    
    print(f"When beta= {beta}, Train_MAP@100= {ap_train}, Train_NDCG@100= {ndcg_train}")
    print(f"When beta= {beta}, Validation_MAP@100= {ap}, Validation_NDCG@100= {ndcg}")
    
spark.stop()
