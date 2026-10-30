from pyspark.sql.types import StructType, StructField, LongType, ArrayType
from map_reduce_friends import *
from spark_init import spark
from als_recommendation import FriendRecommenderALS, get_als_recommendations

def test():
    schema = StructType([
        StructField("UserId", LongType(), False),
        StructField("FriendIds", ArrayType(LongType()), False)
    ])

    data = [
        (1, [2, 3, 4]),
        (2, [1, 5, 6]),
        (3, [1, 5, 7]),
        (4, [1, 2, 6, 8]),
        (5, [2, 3]),
        (6, [2, 4]),
        (7, [3]),
        (8, [4])
    ]

    # Create full df
    df = spark.createDataFrame(data, schema=schema)
    als_df = FriendRecommenderALS.transform_df_to_triples(df)
    recommender = FriendRecommenderALS()

    # 2. Split dataset
    train_df, test_df = als_df.randomSplit([0.8, 0.2])

    print("Randomly split data")

    # 3. Fit with hyperparameter grid search
    best_params = recommender.tune_and_fit(
        train_df=train_df,
    )

    print("Optimal Hyperparameters:", best_params)

    # 4. Evaluate on holdout data
    test_rmse = recommender.evaluate(test_df)
    print(f"Test RMSE: {test_rmse:.4f}")

    # 5. Inspect predictions
    predictions = recommender.predict(test_df)
    predictions.select("User", "OtherUser", "IsFriend", "prediction").show(10)
    random_subset = df.orderBy(rand()).limit(10)
    get_friend_recommendations(df, random_subset)

def main() -> None:
    df = load_friend_network()
    get_als_recommendations(df)
    # als_df = transform_df_to_triples(df)
    # random_subset = df.orderBy(rand()).limit(10)
    # get_friend_recommendations(df, random_subset)

if __name__ == "__main__":
    main()
