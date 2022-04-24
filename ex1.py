import itertools
import sys
import pandas as pd
import numpy as np
from pyspark import SparkContext


# Auxiliar function to create the item baskets and the conditions mappings
def pre_process(data):
    header = data.first()  # extract header
    lines = data.filter(lambda row: row != header) \
                .map(lambda line: tuple(line.split(",")))

    conditions = lines.map(lambda x: (int(x[4]), x[5])) \
        .distinct() \
        .collectAsMap()

    item_baskets = lines.map(lambda x: (x[2], {int(x[4])})) \
                        .reduceByKey(lambda a, b: a | b) \
                        .map(lambda x: x[1])
    return conditions, item_baskets, item_baskets.count()


# Helper function to convert back the codes to the conditions names
def code_to_text(codes):
    if isinstance(codes, tuple):
        return conditions[codes[0]], conditions[codes[1]]
    else:
        return conditions[codes]
    

# Auxiliar function to generate combinations of pairs/tuples
def freq_n_uple(basket, k):
    candidate_n_uple = itertools.combinations(basket, k)
    for n_uple in candidate_n_uple:
        yield(n_uple, 1)


# Receives as input: the baskets
# Returns: frequent items, pairs and triples
def apriori(item_baskets):

    # K = 1
    freqItemCounts = item_baskets.flatMap(lambda x: x) \
                                .map(lambda item: (item, 1)) \
                                .reduceByKey(lambda a, b: a + b) \
                                .filter(lambda item: item[1] >= SUPPORT_THRESHOLD)
    
    # intermediate step                     
    freq_item_count = freqItemCounts.collect()
    freqItemTable = freqItemCounts.map(lambda x: x[0]).collect()
    # Remove the unfrequent items from the baskets
    item_baskets = item_baskets.filter(lambda basket: {item for item in basket if item in freqItemTable})
    
    # K = 2
    pairs = item_baskets.flatMap(lambda x: freq_n_uple(x, 2)) \
                        .reduceByKey(lambda v1, v2: v1 + v2) \
                        .filter(lambda x: x[1] >= SUPPORT_THRESHOLD) \
                        #.sortBy(lambda x: x[1], ascending=False)
    
    # intermediate step                    
    frequent_pairs_count = pairs.collect()
    freq_pair_table = pairs.flatMap(lambda x: x[0]) \
                            .distinct() \
                            .collect()
    # Remove the unfrequent items from the baskets
    item_baskets = item_baskets.filter(lambda basket: {item for item in basket if item in freq_pair_table}) \
        .filter(lambda x: len(x) > 2)
    
    # K = 3
    triples = item_baskets.flatMap(lambda x: freq_n_uple(x, 3)) \
                            .reduceByKey(lambda v1, v2: v1 + v2) \
                            .filter(lambda x: x[1] >= SUPPORT_THRESHOLD) \
                            #.sortBy(lambda x: x[1], ascending=False)
                                                    
    all_supports = freq_item_count + frequent_pairs_count
    
    return frequent_pairs_count, triples, all_supports

# Obtain the association rules
def rule_mining(pairs, triples, all_supports):
    pair_rules = triples.flatMap(lambda triple: [(pair, tuple(set(triple[0]) - set(pair))[0], int(triple[1])) 
                                                 for pair in itertools.combinations(triple[0], 2)]).collect()

    # rules in the form {X} -> {Y} and {Y} -> {X} plus the support of {X,Y}
    single_rules = [[(pair[0][0], pair[0][1], int(pair[1])), (pair[0][1], pair[0][0], int(pair[1]))]
                    for pair in pairs]
    single_rules = list(itertools.chain.from_iterable(single_rules))

    # Auxiliar lists in order to create the dataframes
    all_rules = single_rules + pair_rules
    
    # Auxiliar dataframe to easily map i and j to their respective supports
    supports = pd.DataFrame(data=all_supports, columns=['i', 'support'])
    
    # Dataframe used to calculate the metrics for the rules
    rules = pd.DataFrame(data=all_rules, columns=['i', 'j', 'support_i_j'])

    # Adding the supports of i and j to the rules dataframe
    rules = pd.merge(rules, supports, on=['i'], how='inner')
    rules.rename(columns={'support': 'support_i'}, inplace=True)
    supports.rename(columns={'i': 'j'}, inplace=True)

    rules = pd.merge(rules, supports, on=['j'], how='inner')
    rules.rename(columns={'support': 'support_j'}, inplace=True)
    
    # Converting codes to deseases
    rules['i'] = rules['i'].apply(lambda x: code_to_text(x))
    rules['j'] = rules['j'].apply(lambda x: code_to_text(x))
    
    # Calculating the metrics, more detail in the notebook
    rules['prob_i'] = rules.support_i / total_baskets
    rules['prob_j'] = rules.support_j / total_baskets
    rules['confidence'] = rules.support_i_j / rules.support_i
    rules['interest'] = rules.confidence - rules.prob_j
    rules['lift'] = rules.confidence / rules.prob_j
    rules['aux_calc'] = np.maximum(rules.prob_i + rules.prob_j - 1, 1 / total_baskets) / (rules.prob_i * rules.prob_j)
    rules['std_lift'] = (rules.lift - rules.aux_calc) / ((1 / np.maximum(rules.prob_i, rules.prob_j)) - rules.aux_calc)
    
    # Cleaning the dataframe
    rules.drop(columns=['support_i', 'support_j', 'support_i_j', 'prob_i', 'prob_j', 'aux_calc'], inplace=True)
    rules.sort_values(by=['std_lift'], ascending=False, inplace=True)
    rules.reset_index(inplace=True, drop=True)
    rules = rules.loc[rules.std_lift > 0.2]
    
    return  rules


if __name__ == "__main__":
     
    SUPPORT_THRESHOLD = 1000
    
    try:
        file = sys.argv[1]
    except:
        print("Usage: ex1.py <str:file>")
        exit(1)

    sc = SparkContext(master='local', appName="Assignment1_E1")
    
    original_data = sc.textFile(file)
    
    conditions, item_baskets, total_baskets = pre_process(original_data)
    
    frequent_pairs_count, triples, all_supports = apriori(item_baskets)

    rules = rule_mining(frequent_pairs_count, triples, all_supports)

    rules.to_csv('results/ex_1.csv', index=None)