import pandas as pd
import difflib
import numpy as np
import pickle
import seaborn as sns

from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import SVDpp
from surprise import KNNBasic
from surprise.model_selection import cross_validate

from surprise.model_selection import GridSearchCV

from collections import defaultdict

recipe_data = pd.read_csv('./LAB_2/Lab2_files/RAW_recipes.csv',header=0,sep=",")
print(recipe_data.head())

user_data = pd.read_csv('./LAB_2/Lab2_files/PP_users.csv',header=0,sep=",")
print(user_data.head())

def getRecipeRatings(idx):
    user_items = [int(s) for s in user_data.loc[idx]['items'].replace('[','').replace(']','').replace(',','').split()]
    user_ratings = [float(s) for s in user_data.loc[idx]['ratings'].replace('[','').replace(']','').replace(',','').split()]
    df = pd.DataFrame(list(zip(user_items,user_ratings)),columns = ['Item','Rating'])
    df.insert(loc=0,column='User',value = user_data.loc[idx].u)
    return df

#recipe_ratings = pd.DataFrame(columns = ['User','Item','Rating'])
#for idx,row in user_data.iterrows():
#  recipe_ratings = recipe_ratings.append(getRecipeRatings(row['u']),ignore_index=True)

#recipe_ratings.to_pickle('/work/recipe_ratings.pkl')
recipe_ratings = pd.read_pickle('./LAB_2/Lab2_files/recipe_ratings.pkl')

# sns.barplot(x=recipe_ratings.Rating.value_counts().index, y=recipe_ratings.Rating.value_counts())

recipe_counts = recipe_ratings.groupby(['Item']).size()
filtered_recipes = recipe_counts[recipe_counts>30]
filtered_recipes_list = filtered_recipes.index.tolist()
filtered_recipes_list = filtered_recipes.index.tolist()
print(len(filtered_recipes_list))

recipe_ratings = recipe_ratings[recipe_ratings['Item'].isin(filtered_recipes_list)]

print(recipe_ratings.count())

# sns.barplot(x=recipe_ratings.Rating.value_counts().index, y=recipe_ratings.Rating.value_counts())

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    precision_avg = sum(prec for prec in precisions.values()) / len(precisions)
    recall_avg = sum(rec for rec in recalls.values()) / len(recalls)

    return precision_avg, recall_avg

def mean_average_precision_at_k(predictions, k=10):
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    avg_precisions = []

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        relevant_items = 0
        cumulative_precision = 0.0

        for i, (est, true_r) in enumerate(user_ratings[:k], start=1):
            if true_r >= 3.5:
                relevant_items += 1
                cumulative_precision += relevant_items / i

        if relevant_items > 0:
            avg_precisions.append(cumulative_precision / relevant_items)
        else:
            avg_precisions.append(0)

    return np.mean(avg_precisions)


reader = Reader(rating_scale=(0, 5))

data = Dataset.load_from_df(recipe_ratings[['User', 'Item', 'Rating']], reader)

trainSet = data.build_full_trainset()

algo = NormalPredictor()

cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

anti_testset_user = []
targetUser = 0 #inner_id of the target user
fillValue = trainSet.global_mean
user_item_ratings = trainSet.ur[targetUser]
user_items = [item for (item,_) in (user_item_ratings)]
user_items
ratings = trainSet.all_ratings()
for iid in trainSet.all_items():
  if(iid not in user_items):
    anti_testset_user.append((trainSet.to_raw_uid(targetUser),trainSet.to_raw_iid(iid),fillValue))
    
predictions = algo.test(anti_testset_user)

print(predictions[0])


precision_10, recall_10 = precision_recall_at_k(predictions, k=10)
map_10 = mean_average_precision_at_k(predictions, k=10)
print("Normal Predictor")
print(f'Precision@10: {precision_10}')
print(f'Recall@10: {recall_10}')
print(f'MAP@10: {map_10}')


pred = pd.DataFrame(predictions)
pred.sort_values(by=['est'],inplace=True,ascending = False)
recipe_list = pred.head(10)['iid'].to_list()
print(recipe_data.loc[recipe_list])

# sim_options = {'name': 'cosine',
#                'user_based': False  # compute  similarities between items
#                }
# algo = KNNBasic(sim_options=sim_options)
# # Run 5-fold cross-validation and print results
# cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# predictions = algo.test(anti_testset_user)


# precision_10, recall_10 = precision_recall_at_k(predictions, k=10)
# map_10 = mean_average_precision_at_k(predictions, k=10)
# print(f'Precision@10: {precision_10}')
# print(f'Recall@10: {recall_10}')
# print(f'MAP@10: {map_10}')


# pred = pd.DataFrame(predictions)
# pred.sort_values(by=['est'],inplace=True,ascending = False)
# recipe_list = pred.head(10)['iid'].to_list()
# print(recipe_data.loc[recipe_list])

algo = SVD()
# Run 5-fold cross-validation and print results
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

predictions = algo.test(anti_testset_user)


precision_10, recall_10 = precision_recall_at_k(predictions, k=10)
map_10 = mean_average_precision_at_k(predictions, k=10)
print("SVD")
print(f'Precision@10: {precision_10}')
print(f'Recall@10: {recall_10}')
print(f'MAP@10: {map_10}')


pred = pd.DataFrame(predictions)
pred.sort_values(by=['est'],inplace=True,ascending = False)
recipe_list = pred.head(10)['iid'].to_list()
print(recipe_data.loc[recipe_list])

# param_grid = {'n_factors': [100,150],
#               'n_epochs': [20,25,30],
#               'lr_all':[0.005,0.01,0.1],
#               'reg_all':[0.02,0.05,0.1]}
# grid_search = GridSearchCV(SVD, param_grid, measures=['rmse','mae'], cv=3)
# grid_search.fit(data)

# print(grid_search.best_score['rmse'])
# print(grid_search.best_score['mae'])

# save the model to disk
# pickle.dump(grid_search, open('./Lab2_files/surprise_grid_search_svd.sav', 'wb'))

# load the model from disk
grid_search = pickle.load(open('./LAB_2/Lab2_files/surprise_grid_search_svd.sav', 'rb'))

print(grid_search.best_params['rmse'])

algo = grid_search.best_estimator['rmse']
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)