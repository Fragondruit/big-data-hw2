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
