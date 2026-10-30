from typing import Dict, Any, List, Tuple
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
from spark_init import sc, spark
import joblib

class FriendRecommenderALS:
    """
    ALS-based recommendation system pipeline and tuner for friend networks.
    """

    INIT_SEED: int = 20
    METRIC: str = "rmse"
    PICKLE_PATH: str = "../cache/best_model_weights.pkl"

    def __init__(
            self,
            user_col: str = "User",
            item_col: str = "OtherUser",
            rating_col: str = "IsFriend",
            seed: int = INIT_SEED
    ):
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.seed = seed

        # model containers populated after fitting
        self.pipeline: Pipeline = self._build_pipeline()
        self.cv_model: CrossValidatorModel = None
        self.best_model: PipelineModel = None
        self.evaluator = RegressionEvaluator(
            metricName=FriendRecommenderALS.METRIC,
            labelCol=self.rating_col,
            predictionCol="prediction"
        )

    def _build_pipeline(self) -> Pipeline:
        """Constructs the StringIndexer stages and the ALS model stage."""
        user_indexer = (
            StringIndexer(inputCol=self.user_col, outputCol="UserIndex")
            .setHandleInvalid("keep")
        )

        item_indexer = (
            StringIndexer(inputCol=self.item_col, outputCol="OtherUserIndex")
            .setHandleInvalid("keep")
        )

        als = ALS(
            userCol="UserIndex",
            itemCol="OtherUserIndex",
            ratingCol=self.rating_col,
            coldStartStrategy="drop",
            nonnegative=True,
            seed=self.seed
        )

        return Pipeline(stages=[user_indexer, item_indexer, als])

    def tune_and_fit(
            self,
            train_df: DataFrame,
            num_folds: int = 3,
            ranks: List[int] = [5, 10, 20],
            reg_params: List[float] = [0.01, 0.1],
            max_iters: List[int] = [5, 10],
            parallelism: int = 2
    ) -> Dict[str, Any]:
        """
        Runs cross-validation across the hyperparameter grid and identifies the best model.
        """
        als_stage = self.pipeline.getStages()[-1]
        param_grid = (
            ParamGridBuilder()
            .addGrid(als_stage.rank, ranks)
            .addGrid(als_stage.regParam, reg_params)
            .addGrid(als_stage.maxIter, max_iters)
            .build()
        )

        # CrossValidator with evaluator
        cv = CrossValidator(
            estimator=self.pipeline,
            estimatorParamMaps=param_grid,
            evaluator=self.evaluator,
            numFolds=num_folds,
            seed=self.seed,
            parallelism=parallelism
        )

        # train and get best
        self.cv_model = cv.fit(train_df)
        self.best_model = self.cv_model.bestModel
        joblib.dump(best_model, '../cache/best_model_weights.pkl')
        return self.get_best_params()

    def get_best_params(self) -> Dict[str, Any]:
        """Extracts the winning hyperparameters from the best ALS stage."""
        best_als: ALSModel = self.best_model.stages[-1]
        param_map = best_als.extractParamMap()

        # TODO: use a named object here?
        print({
            "Rank": best_als.rank,
            "RegParam": [0.01, 0.1],
            "MaxIter": [5, 10]
        })

        return {
            "Rank": best_als.rank,
            "RegParam": [0.01, 0.1],
            "MaxIter": [5, 10]
        }

    def predict(self, df: DataFrame) -> DataFrame:
        """Transforms a DataFrame to add the 'prediction' column."""
        return self.best_model.transform(df)

    def evaluate(self, df: DataFrame) -> float:
        """Evaluates model performance against a labeled dataset using RMSE."""
        predictions = self.predict(df)
        return float(self.evaluator.evaluate(predictions))

    @staticmethod
    def transform_df_to_triples(df):
        """Utility to get DF in ALS format."""
        all_user_ids = set(df.select("UserId").rdd.map(lambda r: int(r[0])).collect())
        # broadcast so all executors share the same copy, efficiency purposes
        broadcasted_all_users = spark.sparkContext.broadcast(all_user_ids)
        rdd = df.rdd.flatMap(
            lambda r: list(
                map(
                    lambda other: (
                        int(r["UserId"]),
                        other,
                        1 if other in set(r["FriendIds"] or []) else 0
                    ),
                    filter(lambda u: u != int(r["UserId"]), broadcasted_all_users.value)
                )
            )
        )

        # Convert to DataFrame
        pair_df = rdd.toDF(["User", "OtherUser", "IsFriend"])
        print(pair_df.show(15, truncate=False))
        return pair_df

def get_als_recommendations(df: DataFrame) -> None:
    """ Free utility to do it all. """
    als_df = FriendRecommenderALS.transform_df_to_triples(df)
    recommender = FriendRecommenderALS()
    train_df, test_df = als_df.randomSplit([0.8, 0.2])

    print("[Step 1] Randomly split data.")

    best_params = recommender.tune_and_fit(train_df=train_df)

    print("[Step 2] Optimal Hyperparameters:", best_params)

    test_rmse = recommender.evaluate(test_df)
    print(f"Test RMSE: {test_rmse:.4f}")

    random_subset = (
        test_df.select(col(self.user_col))
        .distinct()
        .orderBy(rand())
        .limit(num_users)
    )

    print("[Step 3] Running on random subset of users...")

    predictions = recommender.predict(random_subset)
    predictions.select("User", "OtherUser", "IsFriend", "prediction").show(10)


