import os
import urllib.request
from math import log
from nltk.stem import PorterStemmer
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.sql.functions import col, concat_ws, regexp_replace
from spark_init import spark

# AG News: 120k news articles, 4 classes, CSV of "label","title","description"
DATA_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
FNAME = "../dataset/ag_news_train.csv"
CLASS_NAMES = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
SEED = 42


def load_news_dataset():
    # download once into the dataset folder (relative to cwd, same as part 1)
    if not os.path.exists(FNAME):
        os.makedirs(os.path.dirname(FNAME), exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, FNAME)

    # fields are quoted, and quotes inside a field are doubled ("")
    df = (
        spark.read.option("header", "false")
        .option("quote", '"')
        .option("escape", '"')
        .csv(FNAME)
        .toDF("Label", "Title", "Description")
    )

    # title + description is the document text
    return df.select(
        col("Label").cast("int"),
        concat_ws(" ", col("Title"), col("Description")).alias("Text")
    )


def preprocess(df):
    # strip html entities like &lt; and #39; so they don't become tokens
    cleaned = df.withColumn("Text", regexp_replace(col("Text"), r"&\w+;|#\d+;", " "))

    # lowercase + split on anything that isn't a letter, then drop stopwords
    tokenizer = RegexTokenizer(inputCol="Text", outputCol="Tokens", pattern="[^a-z]+", minTokenLength=2)
    remover = StopWordsRemover(inputCol="Tokens", outputCol="Filtered")
    filtered = remover.transform(tokenizer.transform(cleaned))

    # stem per partition so each worker only builds one stemmer
    def stem_partition(rows):
        stemmer = PorterStemmer()
        for r in rows:
            yield r["Label"], [stemmer.stem(w) for w in r["Filtered"]]

    return (
        filtered.select("Label", "Filtered").rdd
        .repartition(spark.sparkContext.defaultParallelism)
        .mapPartitions(stem_partition)
    )


def train_naive_bayes(train_rdd):
    num_docs = train_rdd.count()

    # map (label, 1) -> reduce: number of docs per class
    class_counts = train_rdd.map(lambda r: (r[0], 1)).reduceByKey(lambda x, y: x + y).collectAsMap()
    priors = {c: n / num_docs for c, n in class_counts.items()}

    # map ((label, word), 1) for every token -> reduce: count of each word in each class
    word_counts = train_rdd.flatMap(
        lambda r: list(map(lambda w: ((r[0], w), 1), r[1]))
    ).reduceByKey(lambda x, y: x + y).cache()

    # map (label, count) -> reduce: total tokens per class
    class_totals = word_counts.map(lambda kv: (kv[0][0], kv[1])).reduceByKey(lambda x, y: x + y).collectAsMap()
    vocab_size = word_counts.map(lambda kv: kv[0][1]).distinct().count()

    # laplace smoothing: P(w|c) = (count(w, c) + 1) / (total_c + |V|), stored as logs
    denominators = spark.sparkContext.broadcast({c: class_totals[c] + vocab_size for c in class_totals})
    log_likelihoods = (
        word_counts
        .map(lambda kv: (kv[0][1], (kv[0][0], log((kv[1] + 1) / denominators.value[kv[0][0]]))))
        .groupByKey()
        .mapValues(dict)
        .collectAsMap()
    )

    # word seen in training but never in class c -> count is 0, so only the +1 is left
    unseen = {c: log(1 / d) for c, d in denominators.value.items()}

    return {
        "priors": priors,
        "log_likelihoods": log_likelihoods,
        "unseen": unseen,
        "vocab_size": vocab_size,
        "class_totals": class_totals,
    }


def predict(model, rdd):
    # broadcast so all executors share the same copy of the model
    b_model = spark.sparkContext.broadcast(model)

    def classify(tokens):
        m = b_model.value
        scores = {c: log(p) for c, p in m["priors"].items()}
        for w in tokens:
            # words never seen in training carry no information, skip them
            if w in m["log_likelihoods"]:
                for c in scores:
                    scores[c] += m["log_likelihoods"][w].get(c, m["unseen"][c])
        return max(scores, key=scores.get)

    # (actual, predicted)
    return rdd.map(lambda r: (r[0], classify(r[1])))


def evaluate(predictions):
    # map ((actual, predicted), 1) -> reduce: confusion matrix counts
    confusion = predictions.map(lambda p: (p, 1)).reduceByKey(lambda x, y: x + y).collectAsMap()
    total = sum(confusion.values())
    correct = sum(n for (a, p), n in confusion.items() if a == p)

    per_class = {}
    for c in CLASS_NAMES:
        tp = confusion.get((c, c), 0)
        predicted_c = sum(n for (a, p), n in confusion.items() if p == c)
        actual_c = sum(n for (a, p), n in confusion.items() if a == c)
        per_class[c] = (
            tp / predicted_c if predicted_c else 0.0,
            tp / actual_c if actual_c else 0.0,
        )

    return correct / total, per_class, confusion


def main():
    docs = preprocess(load_news_dataset()).cache()
    train_rdd, test_rdd = docs.randomSplit([0.8, 0.2], seed=SEED)
    print(f"Train docs: {train_rdd.count()}  Test docs: {test_rdd.count()}")

    model = train_naive_bayes(train_rdd)
    print(f"Vocabulary size: {model['vocab_size']}")

    print("\nClass priors:")
    for c in sorted(model["priors"]):
        print(f"  {CLASS_NAMES[c]:<10} P(c) = {model['priors'][c]:.4f}  tokens = {model['class_totals'][c]}")

    # most likely words per class, just to sanity check the model
    print("\nTop words per class:")
    for c in sorted(model["priors"]):
        top = sorted(
            model["log_likelihoods"].items(),
            key=lambda kv: -kv[1].get(c, model["unseen"][c])
        )[:10]
        print(f"  {CLASS_NAMES[c]:<10} {', '.join(w for w, _ in top)}")

    accuracy, per_class, confusion = evaluate(predict(model, test_rdd))
    print(f"\nTest accuracy: {accuracy:.4f}")

    print("\nPer class (precision / recall):")
    for c, (precision, recall) in per_class.items():
        print(f"  {CLASS_NAMES[c]:<10} {precision:.4f} / {recall:.4f}")

    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print("  " + " " * 10 + "".join(f"{CLASS_NAMES[c]:>10}" for c in CLASS_NAMES))
    for a in CLASS_NAMES:
        print(f"  {CLASS_NAMES[a]:<10}" + "".join(f"{confusion.get((a, p), 0):>10}" for p in CLASS_NAMES))


if __name__ == "__main__":
    main()
