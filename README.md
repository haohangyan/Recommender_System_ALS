# Project Description
This project involves the implementation of a collaborative-filter based recommender system on a large-scale real-world dataset. The objective is to leverage big data techniques learned during the course to solve a complex problem in music recommendation.

# Dataset
The project utilizes the ListenBrainz dataset, consisting of implicit feedback from music listening behavior. This includes several thousand users and tens of millions of songs, where each observation represents a single interaction between a user and a song.

# Basic Recommender System
Initially, the project requires partitioning the interaction data into training and validation sets. Then a popularity baseline model is built, which should be optimized to perform well on the validation set.

Following the baseline model, a recommendation model using Spark's Alternating Least Squares (ALS) method is implemented to learn latent factor representations for users and items. This model has certain hyper-parameters, such as the rank of latent factors, implicit feedback parameter (alpha), and regularization parameter, which need to be tuned to optimize performance on the validation set.

# Evaluation
After making predictions, the models are evaluated based on their accuracy on the validation and test data. Evaluations are focused on the top 100 items for each user and reported using the ranking metrics provided by Spark. The choice of evaluation criteria for hyper-parameter tuning is left to the discretion of the participants.

# Extensions
Compared the perfomance of Spark ALS with local recommender system coded with LightFm library

