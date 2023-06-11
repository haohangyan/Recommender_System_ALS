from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import dense_rank, col, window,expr
import time
spark = (
    SparkSession.builder.appName("Baseline popularity model with PySpark")
    .master("yarn")
    .config("spark.default.parallelism", 10)  # Adjust this value according to your needs
    .config("spark.submit.deployMode", "client")
    .getOrCreate()
)
# Load the preprocessed data from parquet files
train_interactions_df = spark.read.parquet("/user/hy2664_nyu_edu/final-project-150/train_small.parquet")
test_interactions_df = spark.read.parquet("/user/hy2664_nyu_edu/final-project-150/validation_small.parquet")
train_interactions_df = train_interactions_df.sample(fraction=0.005, seed=42)
test_interactions_df = test_interactions_df.sample(fraction=0.005, seed=42)

# Register DataFrames as temporary tables
train_interactions_df.createOrReplaceTempView("train_interactions")
test_interactions_df.createOrReplaceTempView("test_interactions")

# Encode recording_id using dense_rank function
encoded_train_df = spark.sql("""
    SELECT user_id, dense_rank() OVER (ORDER BY recording_id) as recording_id
    FROM train_interactions
""")
print("finish sql1")
encoded_test_df = spark.sql("""
    SELECT user_id, dense_rank() OVER (ORDER BY recording_id) as recording_id
    FROM test_interactions
""")
print("finish sql2")
# Replace original DataFrames with encoded ones
train_interactions_df = encoded_train_df
test_interactions_df = encoded_test_df


# Compute popularity-based ratings for training set
popularity_rdd = (
    train_interactions_df.groupBy("recording_id", "user_id")
    .count()
    .rdd.map(lambda row: (row.user_id, row.recording_id, row["count"]))
)
print("finish popularity rdd")
popularity_df = spark.createDataFrame(popularity_rdd, ["user_id", "recording_id", "p"])
popularity_df = popularity_df.withColumn("p", popularity_df["p"])
print("finish popularity df")
encoded_train_df.cache()
encoded_test_df.cache()
print("after caching")
popularity_df.cache()
# Obtain ground truth for recommended items
ground_truth = (
    test_interactions_df.select("user_id", "recording_id")
    .groupBy("user_id")
    .agg(expr("collect_set(recording_id) as recording_ids"))
)
ground_truth_rdd = ground_truth.rdd.map(lambda row: (row.user_id, row.recording_ids))
distinct_users_df = test_interactions_df.select("user_id").distinct()
print("finish ground truth")
# Train ALS model
num_user_blocks = 20  # Change this value according to your needs
num_item_blocks = 20  # Change this value according to your needs

for al in [1]:#,10,40,100]:
    for r in [200]:#20, 50,100]:
        for reg in [0.01]:#:1,0.1,0.01,0.001]:
            start_time = time.time()
            als = ALS(
                        maxIter=10,
                        rank=r, 
                        regParam=reg,  
                        userCol="user_id",
                        itemCol="recording_id",
                        ratingCol="p",
                        coldStartStrategy="drop",
                        numUserBlocks=num_user_blocks,
                        numItemBlocks=num_item_blocks,
                        implicitPrefs=True,  # Set to True for implicit feedback
                        alpha=al,  # Set the alpha parameter
                    )
            model = als.fit(popularity_df)
            end_time = time.time()
            print()
            print("finish als, time:",end_time-start_time)
            top_k_recs = model.recommendForUserSubset(distinct_users_df, 100)
            #print("finish recommend")
            top_k_ids = top_k_recs.selectExpr("user_id", "transform(recommendations, x -> x.recording_id) as recording_ids")
            #print("finish forming recoding list")
            top_k_ids_rdd = top_k_ids.rdd.map(lambda row: (row.user_id, row.recording_ids))

            print("finish rdd top k")
            # Join ground truth and recommendations
            joined_rdd = ground_truth_rdd.join(top_k_ids_rdd)
         
            print("finish join")
            # Compute map@100 metric using RankingEvaluator
            metrics = RankingMetrics(joined_rdd.map(lambda x: (x[1][1], x[1][0])))
            ap = metrics.meanAveragePrecisionAt(100)
            print(f"Map@100 = {ap} with alpha = {al}, rank = {r}, regularization= {reg}")
