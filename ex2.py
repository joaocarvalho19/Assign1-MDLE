from pyspark import SparkContext
import re
import random
import numpy as np
import hashlib
import time
import sys
from datetime import datetime

'''
    Receive a document and return a set of shingles 
'''
def shingling(doc, k=6):
    shingles = []
    for i in range(len(doc[1]) - k + 1):
        shingles.append(doc[1][i:i+k])
    
    return (doc[0],set(shingles))

'''
    Given a set `doc_shingles`, pass each member of the set through all permutation
    functions, and set the `ith` position of `vec` to the `ith` permutation
    function's output if that output is smaller than `vec[i]`.
'''
def minhash(doc):
    
    # initialize a minhash of length N with positive infinity values
    doc_shingles = doc[1]
    signature_vector = [float('inf') for i in range(N)]

    for val in doc_shingles:

        # ensure doc_shingles is composed of integers
        if not isinstance(val, int): val = hash(val)

        # loop over each "permutation function"
        for perm_idx, perm_vals in enumerate(perms):
            a, b = perm_vals

            # pass `val` through the `ith` permutation function
            hash_value = ((a * val + b) % LARGE_PRIME)

            # conditionally update the `ith` value of vec
            signature_vector[perm_idx] = min(signature_vector[perm_idx], hash_value)
            #if signature_vector[perm_idx] > hash_value:
                #signature_vector[perm_idx] = hash_value

    # the returned vector represents the minimum hash of the set s
    return doc[0], signature_vector

'''
    Calculate the Jaccard Similarity beetween two sets
'''
def jacc_similarity(a,b):
    a = set(a)
    b = set(b)
    return len(a.intersection(b))/len(a.union(b))

'''
    check if two lists have in common, at least, one element
'''
def common_data(list1, list2):
    result = False
    # traverse in the 1st list
    for x in list1:
  
        # traverse in the 2nd list
        for y in list2:
    
            # if one common
            if sorted(x) == sorted(y):
                result = True
                return result 
            
    return result

'''
    Get similar pairs of docs - "candidates"
'''
def get_candidates(bands, signatures_dict):
    bands_processed = []
    pairs_candidates = []
    docs = list(bands.keys())
    for i1 in range(len(docs)):
        for i2 in range(i1+1, len(docs)):
            band1 = bands[docs[i1]]
            band2 = bands[docs[i2]]
            if common_data(band1, band2):
                sign1 = signatures_dict[docs[i1]]
                sign2 = signatures_dict[docs[i2]]
                jaccard_similarity = jacc_similarity(sign1, sign2)
                pairs_candidates.append((docs[i1],docs[i2],jaccard_similarity))
                    
    return pairs_candidates

'''
    Bands creation
'''
def split_vector(doc, b=20, r=5):
    signature = doc[1]
    # code splitting signature in b parts
    subvecs = []
    for i in range(0, len(signature), r):
        subvecs.append(signature[i : i+r])
    return doc[0], subvecs

''' 
    Given a movie, returns all other movies that are at least 80% similar in terms of their plots but no more than 98%
'''
def similar_movies(movie, similar_candidates, treshold=0.8):
    sim_movies = []
    for m in similar_candidates:
        if (movie == m[0]) and m[2]>0.8 and m[2]<0.98:
            sim_movies.append(m[1])
        if (movie == m[1]) and m[2]>0.8 and m[2]<0.98:
            sim_movies.append(m[0])
            
    return sim_movies

''' 
    Get False Positives rate
'''
def fp_evaluation(shingles_dict, sim_candidates):
    fp = 0
    for cand in sim_candidates:
        doc1, doc2, sim = cand
        shingle1 = shingles_dict[doc1]
        shingle2 = shingles_dict[doc2]
        real_similarity = jacc_similarity(shingle1,shingle2)

        if sim > 0.8 and real_similarity < 0.8: fp+=1

    return (fp/len(sim_candidates))

''' 
    Get False Negatives rate
'''
def fn_evaluation(shingles_dict, sim_candidates):
    fn = 0
    total_candidates = 0
    doc_candidates = []
    docs = list(shingles_dict.keys())
    for id1 in range(len(docs)):
        for id2 in range(id1+1, len(docs)):
            sig1 = shingles_dict[docs[id1]]
            sig2 = shingles_dict[docs[id2]]
            similarity_sig = jacc_similarity(sig1,sig2)
            total_candidates += 1
            if similarity_sig > 0.8: 
                doc_candidates.append((docs[id1],docs[id2]))
    
    false_negatives = len(set(sim_candidates) - set(doc_candidates))
    return (fn/total_candidates)


if __name__ == "__main__":
    file = "assign1/data/small_plot_sum.txt"
    try:
        
        file = sys.argv[1]
        k = int(sys.argv[2])
        r = int(sys.argv[3])
        b = int(sys.argv[4])
        
    except:
        print("Usage: movies.py <int:k> <int:rows> <int:bands>")
        exit(1)
    
    sc = SparkContext(master='local', appName="Assignment1_E2")
    LARGE_PRIME = 4294967311
    
    begin = time.time()
    data = sc.textFile(file)
    item_baskets = data.map(lambda line: re.split('\t', line.lower()))
    
    #Shingling
    docs_shingles = item_baskets.map(lambda line: shingling(line, k))
    #docs_shingles = item_baskets.map(shingling)
    print(docs_shingles.count())
    print("Time elapsed to shingling: ",time.time()-begin)
    
    # Min Hashing
    # specify the length of each minhash vector
    N = 128
    #max_val = (2**32)-1
    max_val = 500

    # create N tuples that will serve as permutation functions
    # these permutation values are used to hash all input sets
    perms = [ (random.randint(0,max_val), random.randint(0,max_val)) for i in range(N)]
    
    # get the signature vectors for each doc
    begin = time.time()
    signatures = docs_shingles.map(minhash)
    print(signatures.count())
    print("Time elapsed to hashing: ",time.time()-begin)
    
    ''' 
    Find as candidates at least 99.5% of pairs with 80% similarity and less than 5% of pairs with 40%
    similarity.
    '''
    begin = time.time()
    signatures_dict = { doc:sign for doc,sign in signatures.collect() }
    bands = signatures.map(lambda line: split_vector(line, b, r))
    bands_dict = { doc:band for doc,band in bands.collect() }
    
    sim_candidates = get_candidates(bands_dict, signatures_dict)
    print("Time elapsed to get candidate pairs: ",time.time()-begin)
    print("Number of Candidates: ",len(sim_candidates))
    
    #Ex: 2.2 - Similar movies
    movie = "31186339"
    sim_movies = similar_movies(movie, sim_candidates)
    print("Movies similar to {}: {}".format(movie, sim_movies))
    
    #Ex: 2.3
    shingles_dict = { doc:shingle for doc,shingle in docs_shingles.collect() }
    
    # FP rate
    if(len(sim_candidates) != 0):
        fp_eval = fp_evaluation(shingles_dict, sim_candidates)
        print("FP eval: ",fp_eval)
    else:
        print("Number of Candidates = 0 !")
    
    # FN rate
    fn_eval = fn_evaluation(shingles_dict, sim_candidates)
    print("FN eval: ",fn_eval)
    
    # Save results
    similar_pairs_rdd = sc.parallelize(sim_candidates).sortBy(lambda line: -line[2])
    format_time = str(datetime.now().strftime("%Y-%m-%dT%H_%M_%S"))
    similar_pairs_rdd.saveAsTextFile("{0}/{1}".format("assign1/results", format_time))
    sc.stop()