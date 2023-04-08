#!/bin/env python
import os
import random
from re import T
import sys
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Source:  https://github.com/maxbachmann/Levenshtein
# Docs:    https://maxbachmann.github.io/Levenshtein/
# Install: mamba install -c conda-forge levenshtein
from rapidfuzz.distance import Indel

DATA_DIR = r'/aiomics/massspec_cache/nist/aiomics/take_two/consensus'
#DATA_DIR = r'.'
INNAME = "2022-10-06_consensus.parquet"
#DATA_DIR = r'.'
#INNAME = "dataset.parquet"
VALID_SIZE = 50000
TEST_SIZE = 50000
SIMILAR_CUTOFF = 0.7

def read_data(columns=None):
    table_name = os.path.join(DATA_DIR, INNAME)
    #table = pq.read_table(table_name, columns=["peptide", "is_phospho"])
    table = pq.read_table(table_name, columns=columns)

    if "is_phospho" not in set(table.column_names):
        print("Creating 'is_phospho' column.")
        mods = table.column("mod_names")
        vals = []
        for mod in mods:
            vals.append(21 in mod.as_py())
        is_phospho = pa.array(vals)

        table = table.append_column("is_phospho", is_phospho)

    print(f"Read {table.num_rows:,} rows from the {INNAME} file.")
    return table

def random_table_split(table):
    indices = list(range(table.num_rows))
    random.shuffle(indices)

    val_idx = indices[:VALID_SIZE]
    test_idx = indices[VALID_SIZE:VALID_SIZE+TEST_SIZE]
    train_idx = indices[VALID_SIZE+TEST_SIZE:]
    
    val = table.take(val_idx)
    test = table.take(test_idx)
    train = table.take(train_idx)

    return(val, test, train)

def random_split(pycol):
    #pycol = col.to_pylist()
    random.shuffle(pycol)

    val = pycol[:VALID_SIZE]
    test = pycol[VALID_SIZE:VALID_SIZE+TEST_SIZE]
    train = pycol[VALID_SIZE+TEST_SIZE:]
    
    return(val, test, train)


def is_similar(query, peps):
    for pep in peps:
        # ld = Levenshtein.distance(query, pep.as_py(), score_cutoff=5)
        ld = Indel.normalized_similarity(query, pep)
        if ld >= SIMILAR_CUTOFF:
            return True
    return False


# The check for this one should always be smaller to larger
def similarity_divide(query, population):
    similar = []
    dissimilar = []
    #dissimilar = query
    for qPep in query:
        if is_similar(qPep, population):
            similar.append(qPep)
        else:
            dissimilar.append(qPep)
    return (similar, dissimilar)

# The check for this one should always be smaller to larger
def similarity_check(query, population):
    #queryPeps = query.column("peptide")
    #popPeps = population.column("peptide")
    similar_list = []
    for qPep in query:
        similar_list.append(is_similar(qPep, population))
    return similar_list

def match_peptides(table, peptides):
    good_peps = set(peptides)
    peps = table.column("peptide").to_pylist()
    mask = []
    for pep in peps:
        if pep in good_peps:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def split_table(table, mask):
    t1 = table.filter(mask)
    return t1
    unmask = [not(i) for i in mask]
    t2 = table.filter(unmask)
    return (t1,t2)

def count_phospho(table):
    #print(table.column("is_phospho").value_counts())
    return table.column("is_phospho").value_counts().field(1)[1].as_py()

def split_by_composition(table):
    #print(table.column("composition").value_counts())
    bestof = set(table.column("peptide").filter(pc.equal(table.column("composition"), "bestof")).unique().to_pylist())
    consensus = set(table.column("peptide").filter(pc.equal(table.column("composition"), "consensus")).unique().to_pylist())
    only_consensus = consensus - bestof
    #print(len(bestof | consensus), len(bestof), len(consensus), len(only_consensus))
    return (only_consensus, bestof)

#table = read_data(columns=["peptide", "composition"])
table = read_data()
#print(table.column_names)
    
peps = table.column("peptide")
uniq_peps = peps.unique()
(only_consensus, bestof) = split_by_composition(table)

print(f"Dataset contains {len(uniq_peps):,} unique peptide strings, {len(only_consensus):,} of which only map to consensus spectra.")

# for pep in peps:
#     print(pep)

valid_peps, test_peps, train_peps = random_split(list(bestof))
train_peps = train_peps + list(only_consensus)

print(f"Checking {len(valid_peps):,} validation peptides against {len(test_peps):,} test peptides.")
valid1_sim, valid1_dis = similarity_divide(valid_peps, test_peps)
print(f"similar: {len(valid1_sim):,}, distinct: {len(valid1_dis):,}")

print(f"Checking {len(valid1_dis):,} validation peptides against {len(train_peps):,} training peptides.")
valid2_sim, valid_dis = similarity_divide(valid1_dis, train_peps)
print(f"similar: {len(valid2_sim):,}, distinct: {len(valid_dis):,}")

print(f"Checking {len(test_peps):,} test peptides against {len(train_peps):,} training peptides.")
test_sim, test_dis = similarity_divide(test_peps, train_peps)
print(f"similar: {len(test_sim):,}, distinct: {len(test_dis):,}")

train_peps = train_peps + valid1_sim + valid2_sim + test_sim
print(f"Training peptide set increased to {len(train_peps):,}")

mask = match_peptides(table, valid_dis)
valid = table.filter(mask)
pq.write_table(valid, "validation.parquet")
print(f"The validation set contains {valid.num_rows:,} rows, {count_phospho(valid):,} of which have phosphorylation.")

mask = match_peptides(table, test_dis)
test = table.filter(mask)
pq.write_table(test, "test.parquet")
print(f"The test set contains {test.num_rows:,} rows, {count_phospho(test):,} of which have phosphorylation.")

mask = match_peptides(table, train_peps)
train = table.filter(mask)
pq.write_table(train, "training.parquet")
print(f"The training set contains {train.num_rows:,} rows, {count_phospho(train):,} of which have phosphorylation.")

sys.exit()


print(f"Checking {valid_peps.num_rows} validation rows against {test_peps.num_rows} test rows.")
sim_mask = similarity_check(valid_peps,test_peps)
valid1_similar, valid1_distinct = split_table(valid_peps, sim_mask)
print(f"{valid1_similar.num_rows} similar rows, {valid1_distinct.num_rows} distinct rows.")

print(f"Checking {valid1_distinct.num_rows} validation rows against {train_peps.num_rows} training rows.")
sim_mask = similarity_check(valid1_distinct,train_peps)
valid2_similar, valid_distinct = split_table(valid1_distinct, sim_mask)
print(f"{valid2_similar.num_rows} similar rows, {valid_distinct.num_rows} distinct rows.")

print(f"Checking {test_peps.num_rows} test rows against {train_peps.num_rows} training rows.")
sim_mask = similarity_check(test_peps, train_peps)
test_similar, test_distinct = split_table(test_peps, sim_mask)
print(f"{test_similar.num_rows} similar rows, {test_distinct.num_rows} distinct rows.")

train_peps = pa.concat_tables([valid1_similar, valid2_similar, test_similar])

pq.write_table(train_peps, "training.parquet")
pq.write_table(test_distinct, "test.parquet")
pq.write_table(valid_distinct, "validation.parquet")

# print(valid.num_rows, valid_similar.num_rows, valid_distinct.num_rows)
# print(valid)
# print(valid_similar)
# print(valid_distinct)

# print(valid.num_rows, test.num_rows, train.num_rows)
# print(valid, test, train)

