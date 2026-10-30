## Setup

- Python 3.12, Java 8/11/17 (Spark 3.5 does not run on newer Java), then:

```
pip install -r requirements.txt
```

All scripts use paths relative to `src/`, so run them from there.

## Running

### Part 2 (Naive Bayes)

```
cd src
python naive_bayes.py
```

The first run downloads the AG News dataset into `dataset/ag_news_train.csv`. The script prints
the class priors, the top words per class, test accuracy, per-class precision/recall, and the
confusion matrix. The pseudo-code and results are in [report_part2.md](report_part2.md).

## Explanations

### Part 1.2

Pseudo-code for algorithm used, written imperatively (using for loops for ease):
```py
result = empty dict()
for user u in dataset:
    votes = empty list()
    
    for f in u.friends:
        S = get f.friends
        T = u.friend 
        W = S - T 
        for person p in W:
            votes.add((p, 1))
        votes.reduceByKey(lambda x, y: x + y)
        
        top_10_recs_for_u = votes.sort().top(10)
        result[u] = top_10_recs_for_u
```

For a fixed user $u$, each friend of a friend $u$ has that is not already my friend, $p$, is ranked by a voting system.
If any of $u$'s friends is friends with $p$, $p$ gets one vote. The more friends $u$ has that also are friends with $p$,
the more popular $p$ is.
