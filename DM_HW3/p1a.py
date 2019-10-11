import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori as apri2
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

store_data = pd.read_csv('p1-a.csv', header=None)
records = []
for i in range(0, 10):
    records.append([str(store_data.values[i,j]) for j in range(1, 6)])

te = TransactionEncoder()
te_ary = te.fit(records).transform(records)
df = pd.DataFrame(te_ary, columns=te.columns_)
df = df.drop(columns='nan')

print(df)

frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

print(frequent_itemsets[ (frequent_itemsets['length'] == 1) & (frequent_itemsets['support'] >= 0.4) ])
print("=====================================")

print(frequent_itemsets[ (frequent_itemsets['length'] == 2) & (frequent_itemsets['support'] >= 0.4) ])
print("=====================================")

print(frequent_itemsets[ (frequent_itemsets['length'] == 2) & (frequent_itemsets['support'] >= 0.7) ])
print("=====================================")

#association_rules = apri2(records, min_support=0.4, min_confidence=1, min_lift=1, min_length=1)
#association_results = list(association_rules)


res = association_rules(frequent_itemsets, metric="confidence", min_threshold=1)
res = res[['antecedents', 'consequents', 'support', 'confidence']]
print(res)
print("=====================================")

res2 = association_rules(frequent_itemsets, metric="confidence", min_threshold=0)
res2 = res2[['antecedents', 'consequents', 'support', 'confidence']]

for i in range(0,15):
    if res2['antecedents'][i]==set('B') and res2['consequents'][i]==set('E'):
        print("confidence of the association rule {B} => {E} is : " + str(res2['confidence'][i]))

#for item in association_results:

    # first index of the inner list
    # Contains base item and add item
    #pair = item[0]
    #items = [x for x in pair]
    #if items[0]!="nan" and items[1]!="nan":
        #print("Rule: " + items[0] + " -> " + items[1])

        #second index of the inner list
        #print("Support: " + str(item[1]))

        #third index of the list located at 0th
        #of the third index of the inner list

        #print("Confidence: " + str(item[2][0][2]))
        #print("Lift: " + str(item[2][0][3]))
        #print("=====================================")