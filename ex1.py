import itertools
import re
from pyspark import SparkContext

SUPPORT_THRESHOLD = 1

sc = SparkContext(master='local', appName="Assignment1_E1")

data = sc.textFile("data/small_conditions.csv")
header = data.first()

baskets = data.filter(lambda row: row != header)\
                .map(lambda line: tuple(line.split(",")))

# Receives as input: the baskets
# Returns: frequent items, pairs and triples
def apriori(item_baskets):
    item_baskets = data.filter(lambda row: row != header)\
                        .map(lambda line: tuple(line.split(",")))
                        
    # Phase 1
    items = item_baskets.flatMap(lambda x: x)
    itemPairs = items.map(lambda item: (item, 1))
    itemCounts = itemPairs.reduceByKey(lambda a, b: a + b)
    freqItemCounts = itemCounts.filter(lambda item: item[1] >= SUPPORT_THRESHOLD)

    # Intermediate Step
    freqItemTable = freqItemCounts.map(lambda x: x[0]).collect()

    # Phase 2
    # k = 2
    pairs = item_baskets.flatMap(lambda x: freq_pairs(x, freqItemTable)) \
                        .reduceByKey(lambda v1, v2: v1 + v2) \
                        .filter(lambda x: x[1] >= SUPPORT_THRESHOLD) \
                        .sortBy(lambda x: x[1], ascending=False)

    # k = 3
    # table of frequent pairs
    frequent_pairs = pairs.map(lambda x: x[0]).collect()
    triples = item_baskets.flatMap(lambda x: freq_triples(x, freqItemTable, frequent_pairs)) \
                        .reduceByKey(lambda v1, v2: v1 + v2) \
                        .filter(lambda x: x[1] >= SUPPORT_THRESHOLD) \
                        .sortBy(lambda x: x[1], ascending=False)
    
    frequent_triples = triples.map(lambda x: x[0]).collect()
    
    return freqItemTable, frequent_pairs, frequent_triples
    
    
# Receives as input: the baskets and the frequent items table
# Returns: candidate frequent pairs
def freq_pairs(basket, table):
    for item_1 in range(0, len(basket)):
        if basket[item_1] not in table:
            continue
        for item_2 in range(item_1 + 1, len(basket)):  # j > i
            if basket[item_2] in table:
                yield(tuple(sorted((basket[item_1], basket[item_2]))), 1)


# Receives as input: the baskets, frequent items and frequent pairs
# Returns: candidate frequent triples
def freq_triples(basket, table, fqt_pairs):
    for item_1 in range(0, len(basket)):
        if basket[item_1] not in table:
            continue
        for item_2 in range(item_1 + 1, len(basket)):  # j > i
            if basket[item_2] not in table:
                continue

            pair = tuple(sorted((basket[item_1], basket[item_2])))
            if pair not in fqt_pairs:
                continue

            for item_3 in range(item_2 + 1, len(basket)):
                if basket[item_3] not in table:
                    continue

                candidate_pairs = list(
                    itertools.combinations((item_1, item_2, item_3), 2))

                # if all candidate pairs are frequent pairs yield the candidate triple
                if all(pair in fqt_pairs for pair in candidate_pairs):
                    yield(tuple(sorted((basket[item_1], basket[item_2], basket[item_3]))), 1)
