# from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, ArrayType
from pyspark.sql.functions import col, split, explode, rand, flatten, collect_list, array_except, array_union, array

def load_friend_network():
    FNAME = "../dataset/soc-LiveJournal1Adj.txt"

    # first, use tab delimeter
    df = (
        spark.read.option("delimiter", "\t")
        .option("header", "false")
        .csv(FNAME)
        .toDF("UserId", "FriendsRaw")
    )

    # then, parse and re-alias the second column
    df_parsed = df.select(
        col("UserId").cast("long"),
        split(col("FriendsRaw"), ",").cast("array<long>").alias("FriendIds"),
    )

    return df_parsed

def transform_df_to_triples(df):
    ...

def get_friend_recommendations(df, random_subset):
    # use RDD API to map-reduce
    rdd = df.select("UserId", "FriendIds").rdd.map(
        lambda r: (int(r["UserId"]), [int(x) for x in r["FriendIds"]])
    )
    subset_rdd = random_subset.select("UserId", "FriendIds").rdd.map(
        lambda r: (int(r["UserId"]), [int(x) for x in r["FriendIds"]])
    )

    # lookup dictionary of candidate direct friends + self like {c: {c, f1, f2, ...}}
    exclusion_map = spark.sparkContext.broadcast(
        dict(subset_rdd.map(lambda r: (r[0], set(r[1]) | {r[0]})).collect())
    )

    # invert subset edges -> key by friend (f, c)
    friend_to_candidate = subset_rdd.flatMap(
        lambda row: list(map(lambda f: (f, row[0]), row[1]))
    )

    # (f, c) with (f, g_list) -> yields (f, (c, [g1, g2, ...]))
    friends_of_friends = friend_to_candidate.join(rdd)

    votes = friends_of_friends.flatMap(
        lambda item: list(
            map(
                lambda g: ((item[1][0], g), 1),
                filter(
                    lambda g: g not in exclusion_map.value.get(item[1][0], set()),
                    item[1][1]
                )
            )
        )
    ).reduceByKey(lambda x, y: x + y)

    top_10_recommendations = ((
        votes
        .map(lambda kv: (kv[0][0], (kv[0][1], kv[1])))
        .groupByKey()
        .mapValues(
            lambda pairs: sorted(list(pairs), key=lambda x: (-x[1], x[0]))[:10]
        )
    )
      .map(lambda kv: (kv[0], list(f for f, v in kv[1])))
      .toDF(["UserId", "RecommendedIds"]))

    # Preview
    print(top_10_recommendations.collect())
    return top_10_recommendations

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
    random_subset = df.orderBy(rand()).limit(10)
    get_friend_recommendations(df, random_subset)

def main() -> None:
    df = load_friend_network()
    random_subset = df.orderBy(rand()).limit(10)
    get_friend_recommendations(df, random_subset)

if __name__ == "__main__":
    # Initialize a SparkSession
    spark = SparkSession.builder \
            .appName("FriendRecommendation") \
            .master("local[*]") \
            .getOrCreate()
    sc = spark.sparkContext
    test()
