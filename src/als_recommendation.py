from typing import Dict, Any, List, Tuple
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame


class FriendRecommenderALS:
    """
    ALS-based recommendation system pipeline and tuner for friend networks.
    """

    INIT_SEED: int = 20
    METRIC: str = "rmse"

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
            metricName=METRIC,
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

        # 3. ParamGridBuilder varying rank, regParam, and maxIter
        param_grid = (
            ParamGridBuilder()
            .addGrid(als_stage.rank, ranks)
            .addGrid(als_stage.regParam, reg_params)
            .addGrid(als_stage.maxIter, max_iters)
            .build()
        )

        # 4. CrossValidator with evaluator
        cv = CrossValidator(
            estimator=self.pipeline,
            estimatorParamMaps=param_grid,
            evaluator=self.evaluator,
            numFolds=num_folds,
            seed=self.seed,
            parallelism=parallelism
        )

        # Execute tuning
        self.cv_model = cv.fit(train_df)
        self.best_model = self.cv_model.bestModel

        return self.get_best_params()

    def get_best_params(self) -> Dict[str, Any]:
        """Extracts the winning hyperparameters from the best ALS stage."""
        best_als: ALSModel = self.best_model.stages[-1]
        return {
            "Rank": best_als.getRank(),
            "RegParam": best_als.getRegParam(),
            "MaxIter": best_als.getMaxIter()
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
