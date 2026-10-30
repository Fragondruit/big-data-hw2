from pyspark.sql.types import StructType, StructField, LongType, ArrayType
from map_reduce_friends import *
from spark_init import spark

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
    # als_df = transform_df_to_triples(df)
    random_subset = df.orderBy(rand()).limit(10)
    get_friend_recommendations(df, random_subset)

def main() -> None:
    df = load_friend_network()
    als_df = transform_df_to_triples(df)
    random_subset = df.orderBy(rand()).limit(10)
    # get_friend_recommendations(df, random_subset)

if __name__ == "__main__":
    test()
