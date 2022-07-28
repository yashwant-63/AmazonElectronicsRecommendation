#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#data preparation
#Mounting to google drive
from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
#Setting path
path = '/content/drive/MyDrive/CSE6240/Project' 
# %cd $path

get_ipython().system('pwd')

#Reading File
import pandas as pd
ratings = pd.read_csv("ratings_Electronics.csv",names=['user', 'item','comma_separated_list_of_features','timestamp'])
ratings["state_label"]=0

#ratings["timestamp"]=(ratings["timestamp"]-ratings["timestamp"].mean())/ratings["timestamp"].std()

#Checking Ratings
ratings.head()

#Creating unique index for users
unique_user=ratings["user"].unique()
unique_user.sort()
user_index={unique_user[i]:i for i in range(len(unique_user))}

#Creating unique index for items 
unique_item=ratings["item"].unique()
unique_item.sort()
item_index={unique_item[i]:i for i in range(len(unique_item))}
user_index=pd.DataFrame(user_index.items(),columns=["User_Id","User"])
item_index=pd.DataFrame(item_index.items(),columns=["Item_Id","Item"])

#Joined to replace item and user value with index
ratings=ratings.merge(user_index,left_on="user",right_on="User_Id",how="inner")
ratings=ratings.merge(item_index,left_on="item",right_on="Item_Id",how="inner")
ratings=ratings[['User', 'Item', 'timestamp', 'state_label',
            'comma_separated_list_of_features']]
ratings=ratings.rename(columns={"User": "user", "Item": "item"})

#Finding count of interactions per item /user
user_count=ratings[["user","comma_separated_list_of_features"]].groupby("user").count().reset_index()
item_count=ratings[["item","comma_separated_list_of_features"]].groupby("item").count().reset_index()

#Sorting item count
item_count=item_count.sort_values(by="comma_separated_list_of_features", ascending=False)

item_count=item_count.reset_index()

#Sort values by count
user_count=user_count.sort_values(by="comma_separated_list_of_features", ascending=False)
user_count=user_count.reset_index()

user_index.to_csv("/content/drive/MyDrive/CSE6240/Project/data/user_index.csv",index=False)
item_index.to_csv("/content/drive/MyDrive/CSE6240/Project/data/item_index.csv",index=False)

#Select top users and items
item_list=item_count[:1000]["item"]
user_list=user_count[:8000]["user"]
filter2=ratings["item"].isin(item_list)
filter1=ratings["user"].isin(user_list)
ratings_filtered=ratings[filter1&filter2]
#ratings_filtered.loc["timestamp"]=ratings_filtered["timestamp"]-ratings_filtered["timestamp"].min()


#Sorted dataset by timestamp and substracted the minimum timestamp

ratings_filtered=ratings_filtered.sort_values(by="timestamp")
min1=ratings_filtered["timestamp"].min()
ratings_filtered["timestamp"]=ratings_filtered["timestamp"]-min1

#saving file

ratings_filtered.to_csv("/content/drive/MyDrive/CSE6240/Project/data/ratings_filtered_top_1000_top_8k_user.csv",index=False)



#CTDNE Implementation


# In[1]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from operator import itemgetter as it
from itertools import repeat
import pandasql as ps
from numpy import dot
from numpy.linalg import norm
from stellargraph import StellarGraph
from stellargraph.datasets import IAEnronEmployees

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


am_df = pd.read_csv('ratings_filtered_top_1000_top_8k_user.csv')
am_df.head()


# In[4]:


am_df_st = am_df[['user','item','timestamp']]
am_df_st.rename(columns = {'user':'source' , 'item':'target','timestamp':'time'} , inplace = True)


# In[7]:


am_df_st['source'] = am_df_st['source'].astype(str)
am_df_st['target'] = am_df_st['target'].astype(str)


# In[20]:
trn_ind = int(0.8*len(am_df_st))
val_ind = int(0.1*len(am_df_st))

am_df_st_tr = am_df_st[:trn_ind]
am_df_st_val = am_df_st[trn_ind:trn_ind+val_ind]
am_df_st_tst = am_df_st[trn_ind+val_ind:]


graph = StellarGraph(
    edges=am_df_st,
    edge_weight_column="time",
)


# In[21]:


num_walks_per_node = 10
walk_length = 80
context_window_size = 10


# In[22]:


num_cw = len(graph.nodes()) * num_walks_per_node * (walk_length - context_window_size + 1)


# In[23]:


from stellargraph.data import TemporalRandomWalk

temporal_rw = TemporalRandomWalk(graph)
temporal_walks = temporal_rw.run(
    num_cw=num_cw,
    cw_size=context_window_size,
    max_walk_length=walk_length,
    walk_bias="exponential",
)

print("Number of temporal random walks: {}".format(len(temporal_walks)))


# In[26]:


from gensim.models import Word2Vec

embedding_size = 128
temporal_model = Word2Vec(
    temporal_walks,
    vector_size=embedding_size,
    window=context_window_size,
    min_count=0,
    sg=1,
    workers=2,
)


# In[44]:


unseen_node_embedding = np.zeros(embedding_size)
def temporal_embedding(u):
    try:
        return temporal_model.wv[u]
    except KeyError:
        print("Not found embedding for : ",u)
        return unseen_node_embedding


# In[28]:


temporal_node_embeddings = temporal_model.wv.vectors


# In[38]:


emb_df = pd.DataFrame(temporal_node_embeddings)


# In[45]:


user_ratings = {}
def user_id_embeddings():
    for user in am_df_st['source']:
        user_ratings[str(user)] = temporal_embedding(str(user))
user_id_embeddings()


# In[48]:


item_ratings = {}
def item_id_embeddings():
    for item in am_df_st['target']:
        item_ratings[str(item)] = temporal_embedding(str(item))
item_id_embeddings()


# In[62]:


pd.DataFrame(user_ratings).to_csv('user_embeddings.csv')


# In[63]:


pd.DataFrame(item_ratings).to_csv('item_embeddings.csv')


# In[55]:


def cos_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))


# In[ ]:


user_list = [u for u in user_ratings.keys()]
item_list = [i for i in item_ratings.keys()]


cosine_sim_df = pd.DataFrame(columns = user_list)
cosine_sim_df['index'] = item_list
cosine_sim_df.index = item_list
cosine_sim_df = cosine_sim_df.drop(['index'],axis = 1)

for k1,v1 in user_ratings.items():
    
    for k2,v2 in item_ratings.items():
        
        cosine_sim_df[k1][k2]=(cos_sim(v1,v2))
    


# In[ ]:


temp1 = cosine_sim_df.transpose()
temp2 = temp1.reset_index()


# In[ ]:


def find_top_k ( cosine_sim_df,k = 10 ):
    list_of_vals = []
    n = 10
    new_d = (zip(repeat(row["index"]), map(it(0),(row[1:].sort_values(ascending=False)[:n].iteritems())))
                     for _, row in cosine_sim_df.iterrows())
    for row in new_d:
        for ele in row:
            list_of_vals.append(ele)
    return pd.DataFrame(list_of_vals , columns = ['User','Pred_Item'])


# In[ ]:


recommended_items = find_top_k(  temp2,k = 10)


# In[ ]:


new_df = pd.merge(am_df_st, recommended_items,  how='inner', left_on=['source','target'], right_on = ['User','Pred_Item'])


# In[ ]:


rec_items = ps.sqldf(""" Select User,Pred_Item, row_number() over (partition by User) as rank 
                        from  recommended_items  """)
rec_item2 = rec_items.drop_duplicates(subset=['User','Pred_Item'])


# In[ ]:


mrr_df = pd.merge(rec_item2, am_df_st,  how='inner', left_on=['User','Pred_Item'], right_on = ['source','target'])
mrr_temp1 = ps.sqldf(""" Select avg(mrr) as avg_mrr from
                        (Select User, 1/rank as mrr from mrr_df)""")


# In[ ]:


recall_10_df = ps.sqldf("""   Select avg(num)/10 as recall_10 from 
                    (Select source,count(*) as num from new_df 
                      group by 1)  """)

