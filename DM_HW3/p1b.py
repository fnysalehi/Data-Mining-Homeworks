import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori as apri2
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

store_data = pd.read_csv('p1-b.csv', header=None)

records = []
for i in range(0, 4):
    records.append([str(store_data.values[i,j]) for j in range(1, 5)])
te = TransactionEncoder()
te_ary = te.fit(records).transform(records)
df = pd.DataFrame(te_ary, columns=te.columns_)
df = df.drop(columns='nan')
print (df)
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets)
print("=====================================")


res = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.65)
res = res[['antecedents', 'consequents', 'support', 'confidence']]
print(res)
print("=====================================")


res2 = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)
res2 = res2[['antecedents', 'consequents', 'support', 'confidence']]
print(res2)
print("=====================================")

for i in range(0,13):
    if res['antecedents'][i]==set('E') and res['consequents'][i]==set('C'):
        print("confidence of the association rule {E} => {C} is : " + str(res['confidence'][i]))
    if res['antecedents'][i] == set('B') and res['consequents'][i] == set('C'):
        print("support value of the association rule {B} => {C} is : " + str(res['support'][i]))

