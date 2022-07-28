
#Mounting to google drive
from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
#Setting path
path = '/content/drive/MyDrive/CSE6240/Project' 
# %cd $path

!pwd

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

