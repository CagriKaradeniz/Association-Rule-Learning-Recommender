
import pandas as pd
import numpy as np
import Utils_cagri as util
from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option('display.max_columns', 20)
pd.set_option('display.expand_frame_repr', False)
#loading data
df=pd.read_excel("online_retail_II.xlsx",sheet_name="Year 2010-2011")
data=df.copy()


data=util.data_hazirlama(data)
data.shape
data.head()
data.describe().T
util.dataset_ozet(data)

#Selecting German customers

data_ger=data[data["Country"]=="Germany"]
data_ger.shape
data_ger.head()
data_ger.describe().T
util.dataset_ozet(data_ger)

ger_rules=util.kural_olustur(data,country="Germany")

#Analyzing random three user

#Kullanıcı 1 ürün id'si: 21987
#Kullanıcı 2 ürün id'si: 23235
#Kullanıcı 3 ürün id'si: 22747
id_list=[21987,23235,22747]
[util.check_id(data_ger,i) for i in id_list]



#Recommendation with ARL
for k in id_list:
    rec=util.arl_recommender(ger_rules,k,2)
    print(f"{k} İçin recommended list: {rec}")

for k in id_list:
    rec=util.arl_recommender(ger_rules,k,2)
    print(f"{util.check_id(data_ger,k)} İçin recommended list: {[util.check_id(data_ger,i) for i in rec]}")

