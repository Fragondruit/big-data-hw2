from pyspark.sql.functions import col, split, coalesce, rand, array, when
from spark_init import spark

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
        coalesce(
            when(col("FriendsRaw").isNotNull() & (col("FriendsRaw") != ""),
                 split(col("FriendsRaw"), ",").cast("array<long>")),
            array().cast("array<long>")
        ).alias("FriendIds")
    )

    return df_parsed


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

    # Output format: UserID<TAB>rec1,rec2,...
    for row in top_10_recommendations.orderBy("UserId").collect():
        print(f"{row['UserId']}\t{','.join(str(r) for r in row['RecommendedIds'])}")
    return top_10_recommendations

