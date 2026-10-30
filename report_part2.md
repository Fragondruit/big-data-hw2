# Part 2: Naive Bayes Classifier

## Dataset

AG News: 120,000 news articles in 4 classes (World, Sports, Business, Sci/Tech).
Public URL: https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv

## Preprocessing

1. Lowercase and tokenize on non-letters
2. Remove stopwords
3. Porter stemming
4. 80/20 train/test split (seed 42)

## Pseudo-code

```
# training
map:    (label, tokens)       -> (label, 1)
reduce: sum                   -> docs per class N_c,  prior P(c) = N_c / N

map:    (label, tokens)       -> ((label, w), 1) for each word w
reduce: sum                   -> count(w, c)

map:    ((label, w), count)   -> (label, count)
reduce: sum                   -> total words per class

P(w|c) = (count(w, c) + 1) / (total words in c + vocab size)

# testing
map:    (label, tokens)       -> (label, argmax_c [log P(c) + sum of log P(w|c)])
map:    (label, predicted)    -> 1 if correct else 0
reduce: sum / total           -> accuracy
```

## Results

| Class    | Prior  |
|----------|--------|
| World    | 0.2491 |
| Sports   | 0.2504 |
| Business | 0.2500 |
| Sci/Tech | 0.2504 |

**Test accuracy: 90.44%**

Sports was the easiest class to predict. Most errors were between Business and Sci/Tech, since tech
company news uses words from both.
