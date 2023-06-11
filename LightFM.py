import pandas as pd
import numpy as np
import time 
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k

df_train = pd.read_parquet('/content/train_small.parquet')
df_val = pd.read_parquet('/content/val_small.parquet')
df_test = pd.read_parquet('/content/test.parquet')

# Downsample
df_train = df_train.sample(frac=0.01)
df_val = df_val.sample(frac=0.01)
df_test = df_test.sample(frac=0.01)

df = pd.concat([df_train, df_val, df_test])
user_id = df['user_id'].unique()
recording_id = df['recording_id'].unique()

data = Dataset()

data.fit(users=user_id,
         items=recording_id
        )

# Build interactions matrix
(interactions_train, weights_train) = data.build_interactions(((x['user_id'], x['recording_id']) for index, x in df_train.iterrows()))
(interactions_val, weights_val) = data.build_interactions(((x['user_id'], x['recording_id']) for index, x in df_val.iterrows()))
(interactions_test, weights_test) = data.build_interactions(((x['user_id'], x['recording_id']) for index, x in df_test.iterrows()))

start_time = time.time()
model = LightFM(loss='warp')
model.fit(interactions_train, epochs=50, num_threads=1)
end_time = time.time()
print("Total LightFM model fitting time: {} seconds".format(end_time - start_time))

training_precision = precision_at_k(model, interactions_train, k=100).mean()
print("Training Precision: {}".format(training_precision))
validation_pricision = precision_at_k(model, interactions_val, k=100).mean()
print("Validation Precision: {}".format(training_precision))

# Evaluation
test_precision = precision_at_k(model, interactions_test, k=100).mean()
print("Testing Precision: {}".format(test_precision))
