# %%
# Imports
import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
from math import pi
import catboost as cb
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

# def similarity(impMovieRatingsList, movieNum, impMovieRatingsList2, movieNum2):
#     '''Based on the given list of user movie ratings of movie, looks at how similar each users ratings is to movie2.
#     The closer the ratings are, the higher the similarity score.'''

#     d = 0

#     for i in range(len(impMovieRatingsList.columns)):

#         # print(i)
#         # print((float(impMovieRatingsList.iloc[movieNum, i]))) # Check
#         # print(float(impMovieRatingsList2.iloc[movieNum2, i])) # Check

#         d += ((float(impMovieRatingsList.iloc[movieNum, i]) - float(impMovieRatingsList2.iloc[movieNum2, i]))**2)

# #     print(d) # Check

#     return(1/(1+math.sqrt(d)))

# def recFilm(movieRatingsList, user):
#     '''Provides a sorted list of the highest predicted ratings for films for a given user by comparing
#     other user ratings for films the user hasn't seen to those same users ratings on films the user has seen
#     and providing a similarity score. This score is then used to calculate a predicted score for each of the
#     films the user hasn't seen, which is then sorted and returned.'''

#     # calcCount = 0

#     # Drop films that no user has rated.
#     cols = list(movieRatingsList.columns[11:])
#     impMovieRatingsList = movieRatingsList.dropna(axis=0, how='all', subset = cols)#.reset_index(drop = True)


#     # Drop all irrelevant columns - leaving only film name, IMDB rating and user ratings
#     impMovieRatingsList = impMovieRatingsList.drop(["tconst", "numVotes", "titleType", "startYear","runtimeMinutes", "genre1", "primaryTitle", "genre2", "genre3", "NoUserInput"], 1)
#     # print(impMovieRatingsList)
#     # Create a new dataframe with only the films that the user has rated
#     impMovieRatingsList2 = impMovieRatingsList.dropna(subset=[user])
#     # print(impMovieRatingsList2)
#     # Want a df with just ratings
#     impMovieRatingsList3 = impMovieRatingsList
#     # print(impMovieRatingsList3)
#     # Cut down the original dataframe to only films the user hasnt rated
#     impMovieRatingsList = impMovieRatingsList[impMovieRatingsList[user].isna()]
#     # print(impMovieRatingsList)
#     # Create a list that is a mean of all the films' ratings
#     m = impMovieRatingsList3.mean(axis=1)
#     # print(m)

#     # Imputation - for each user
#     for col in range(len(impMovieRatingsList3.columns)):
#         ratingCount = 0
#         diffSum = 0

#         # For each film that has a rating add the difference between the mean rating of that film and the user's rating of that film.
#         i = 0
#         for row in impMovieRatingsList3.index:
#             # calcCount += 1
#             if not math.isnan(impMovieRatingsList3.iloc[i,col]):
#                 ratingCount += 1
#                 diffSum += impMovieRatingsList3.iloc[i,col] - m[row]
#             i += 1

#         # Calculate the average difference between the user's rating and the average rating.
#         finalDiff = diffSum/ratingCount
#         # print(finalDiff)

#         # For each film that the user has not rated, replace the NA value with
#         # the mean of that film plus average difference between the user's
#         # rating and the average rating.
#         i = 0
#         for row in impMovieRatingsList3.index:
#             # calcCount += 1
#             if math.isnan(impMovieRatingsList3.iloc[i,col]):
#                 impMovieRatingsList3.iloc[i,col] = m[row] + finalDiff
#             i += 1

#     # Assign the imputed columns to the imputed dataframes.
#     for col in impMovieRatingsList3.columns:
#         # calcCount += 1
#         impMovieRatingsList[col] = impMovieRatingsList3[col]
#         impMovieRatingsList2[col] = impMovieRatingsList3[col]


#     # Drop the user from both dataframes
#     impMovieRatingsList = impMovieRatingsList.drop(user, 1)
#     impMovieRatingsList2 = impMovieRatingsList2.drop(user, 1)


#     # print(impMovieRatingsList) # Check
#     # print(impMovieRatingsList2) # Check

#     finalRatings = []
#     impMovieCount = 0

#     # For each movie that the user hasn't rated...
#     for movieNum in impMovieRatingsList.index:
#         simList = []
#         impMovieCount2 = 0

#         # For each movie that the user has rated...
#         for movieNum2 in impMovieRatingsList2.index:
#             # calcCount += 1

#             # If the movies aren't the same...which they shouldn't be
#             if movieNum != movieNum2:

#                 # Create a tuple of how similar the film they haven't seen is with the film they have seen based on all users ratings.
#                 myTup = (str(similarity(impMovieRatingsList, impMovieCount, impMovieRatingsList2, impMovieCount2)),str(movieNum), str(movieNum2))
#                 simList.append(myTup)

#             # Do this for all films the user has rated.
#             impMovieCount2 += 1
#         # print(simList)
#         # Pick the top 5 films the user has seen that are most similar (according to all other user ratings) to the film the user hasn't seen
#         top5 = sorted(simList, reverse = True)[:5]

#         # print(top5) # Check 1

#         numerator = 0
#         denominator = 0

#         # For each film in the top 5....
#         for i in top5:
#             # calcCount += 1

#             # print(movieRatingsList.loc[int(i[1]), user]) # Check 2 - Should be NaNs
# #             print(float(movieRatingsList.loc[int(i[2]), user])) # Check 1 - check against movieRatingsList

#             # The numerator is the sum of each of the ratings the user has given for those films multiplied by the similarity score (for weighting)
#             numerator += float(movieRatingsList.loc[int(i[2]), user]) * float(i[0])

#             # The denominator is the sum of the top 5 similarity scores (for weighting)
#             denominator += float(i[0])

#         # The predicted rating for that film
#         rating = numerator/denominator

# #         print(rating) # Check 1

#         # A tuple of the rating and the film's index.
#         newTup = (rating, movieRatingsList.iloc[movieNum, 0])

# #         print(newTup) # Check 1

#         # Add both to a list of ratings, from which the recommendation will be picked
#         finalRatings.append(newTup)

#         impMovieCount += 1

# #     print(finalRatings) # Check 3
#     # print(calcCount)
#     # print(len(finalRatings))
#     # Return the sorted list of the highest predicted rated films for the user.
#     return(sorted(finalRatings, reverse = True))

# First step is to infill the "entire" ratings table - just the ones that have 
# been rated by someone
# folder_name = 'C:/Users/lwals/OneDrive/Desktop/CodingProjects/MovieRecommender'
# folder_path = Path(folder_name)
# file_name = 'movieRatingsList_20230616.csv'

# df = pd.read_csv(folder_path / file_name)

# df = df.set_index('tconst')

# detail_df = df[['titleType', 'primaryTitle']]

# drop_cols = ['titleType', 'primaryTitle']
# df = df.drop(drop_cols, axis=1)

def infill_missing_ratings(df, user_cols: list, avg_col = 'averageRating'):
    df_w = df.copy()
    cols =  [avg_col] + user_cols
    # First want to get only the films that someone has rated
    df_w = df_w[cols].dropna(subset=user_cols, how='all').reset_index(drop=True)
    # Imputing those misssing ratings
    imp = IterativeImputer(max_iter=20, random_state=0, min_value=0, max_value=10)
    imp.fit(df_w.iloc[:,1:])
    imp_df_w = pd.DataFrame(imp.transform(df_w.iloc[:,1:]))

    # Assigning the correct column names
    imp_df_w.columns = df_w.iloc[:,1:].columns

    # Adding back in the ID column to rejoin the rest of the data

    # Join back the rest of the columns
    df_w = df.copy().reset_index(drop=False).drop(cols, axis=1)

    final_df = imp_df_w.merge(
        df_w, 
        how='inner', 
        right_index=True, 
        left_index=True)
    
    final_df = final_df.set_index('tconst')

    return final_df

# imp_df = infill_missing_ratings(df, list(df.iloc[:,8:].columns))

def encode_and_combine_columns(df, encode_cols: list):

    df_w = df.copy()

    full_list = []

    for col in encode_cols:
        full_list = full_list + list(df_w[col])

    unique_list = set(full_list) 
    unique_list.remove(np.nan)

    for col in encode_cols:
        df_w = pd.get_dummies(df_w, columns=[col], prefix=col)

    for val in unique_list:
        df_w[val] = 0
        for col in encode_cols:
            if col + '_' + val in df_w.columns:
                df_w[val] = df_w[val] + df_w[col + '_' + val]

    cols_to_drop = []

    for col in encode_cols:
        cols_to_drop = cols_to_drop + list(df_w.filter(like=col).columns)

    df_w.drop(cols_to_drop, axis=1, inplace=True)


    return df_w

def build_genre_ratings_df(df, user_list, spider = True):

    out_df = pd.DataFrame()

    if spider:
        genre_list = ['Comedy', 'Drama', 'Action', 'Crime', 'Adventure', 'Horror']

    else:
        genre_list = list(set(list(df['genre1'].unique()) + list(df['genre2'].unique()) + list(df['genre3'].unique())))
        genre_list.remove(np.nan)

    # Overall
    out_df['user'] = ['average']
    for i in genre_list:
        out_df[i] = [np.nanmean(df[
            ((df['genre1'] == i) | (df['genre2'] == i) | (df['genre3'] == i)) & 
            (df['NoUserInput'] == False)].iloc[:,8:])]

    for u in user_list:
        user_scores = [u]
        for i in genre_list:
            df[
                ((df['genre1'] == i) | (df['genre2'] == i) | (df['genre3'] == i)) & 
                (df['NoUserInput'] == False)][u].mean()
            user_scores.append(
                df[
                    ((df['genre1'] == i) | (df['genre2'] == i) | (df['genre3'] == i)) & 
                    (df['NoUserInput'] == False)][u].mean())
        out_df.loc[len(out_df)] = user_scores

    return out_df.fillna(0)



def genre_spider_chart(data, save_loc):

    # number of variable
    categories = list(data)[1:]
    N = len(categories)
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
    
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axe per variable + add labels with increased font size
    plt.xticks(angles[:-1], categories, fontsize=12)
    ax.xaxis.set_tick_params(pad=10)
    
    # Draw ylabels with increased font size
    ax.set_rlabel_position(0)
    plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", '8', '10'], color="grey", size=12)
    plt.ylim(0, 10)

    # Ind2
    values = data.loc[1].drop('user').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=3, linestyle='solid', label=data.loc[1, 'user'], color='darkred')
    # ax.fill(angles, values, color='darkred', alpha=0.1)

    # Ind1
    values = data.loc[0].drop('user').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label="LMR Average", color='darkgrey')
    # ax.fill(angles, values, 'darkgrey', alpha=0.1)
    
    # Ind2
    if data.shape[0] == 3:
        values = data.loc[2].drop('user').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=data.loc[2, 'user'], color='#607D8B')
        # ax.fill(angles, values, '#607D8B', alpha=0.1)

    # Add legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=data.shape[0], fontsize=14)

    plt.savefig(save_loc + '/spider_chart.png', dpi=500, bbox_inches='tight')

def top_5_genres(df, user):

    genre_ratings = build_genre_ratings_df(df, [user], False)

    genre_ratings.set_index('user', inplace = True)

    top_5 = genre_ratings.loc[user].sort_values(ascending=False)[:5]

    return top_5

def most_sim_user(df, user):

    user_df = df[df[user].notnull()].iloc[:,8:]

    other_users = list(user_df.columns)
    other_users.remove(user)

    min = 10

    for u in other_users:
        shared_df = user_df[user_df[u].notnull()][[user,u]]
        if len(shared_df) > 0:
            shared_df['diff_sq'] = shared_df.apply(lambda row: (row[user] - row[u])**2, axis=1)
            msd = shared_df['diff_sq'].mean()
            if msd < min:
                min = msd
                most_sim = u

    return most_sim

# %%

# Ready to model - let't try naive model on Linus



# Table for scores to be added

pred_df = df[df['NoUserInput'] == False].iloc[:,:7]

model_dict = {}
feature_dict = {}

for user in final_df.columns[:-24]:


    trial_df = final_df[final_df.index.isin(df[df[user].notnull()].index)]


    y = trial_df[user]
    X = trial_df.drop([user], axis=1)

    res_dict = {}
    min_rmse = 100

    for i in range(5, 15):

        y = trial_df[user]
        X = trial_df.drop([user], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size = 0.2, 
            random_state=8)

        train_dataset = cb.Pool(X_train, y_train) 
        test_dataset = cb.Pool(X_test, y_test)

        model = cb.CatBoostRegressor(
            loss_function='RMSE',
            iterations=200,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=0.5)

        sf = model.select_features(train_dataset,
                            features_for_select=X.columns, 
                            num_features_to_select=i,
                            verbose=100)

        X = trial_df[sf['selected_features_names']]

        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size = 0.2, 
            random_state=8)

        train_dataset = cb.Pool(X_train, y_train) 
        test_dataset = cb.Pool(X_test, y_test)

        model = cb.CatBoostRegressor(loss_function='RMSE')

        grid = {'iterations': [100, 150, 200],
                'learning_rate': [0.03, 0.1],
                'depth': [2, 4, 6, 8],
                'l2_leaf_reg': [0.2, 0.5, 1, 3]}
        model.grid_search(grid, train_dataset, verbose=100)



        pred = model.predict(X_test)
        rmse = (np.sqrt(mean_squared_error(y_test, pred)))
        r2 = r2_score(y_test, pred)

        if rmse < min_rmse:
            chosen_num_features = i
            min_rmse = rmse

        res_dict[i] = (rmse, r2)



    y = trial_df[user]
    X = trial_df.drop([user], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size = 0.2, 
        random_state=8)

    train_dataset = cb.Pool(X_train, y_train) 
    test_dataset = cb.Pool(X_test, y_test)

    model = cb.CatBoostRegressor(
        loss_function='RMSE',
        iterations=200,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=0.5)

    sf = model.select_features(
        train_dataset,
        features_for_select=X.columns, 
        num_features_to_select=chosen_num_features,
        verbose=100)

    X = trial_df[sf['selected_features_names']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size = 0.2, 
        random_state=8)

    train_dataset = cb.Pool(X_train, y_train) 
    test_dataset = cb.Pool(X_test, y_test)

    model = cb.CatBoostRegressor(loss_function='RMSE')

    grid = {
        'iterations': [100, 150, 200],
        'learning_rate': [0.03, 0.1],
        'depth': [2, 4, 6, 8],
        'l2_leaf_reg': [0.2, 0.5, 1, 3]}

    model.grid_search(grid, train_dataset, verbose=100)




    # print('Testing performance')
    # print('RMSE: {:.2f}'.format(rmse))
    # print('R2: {:.2f}'.format(r2))


      
# sorted_feature_importance = model.feature_importances_.argsort()
# plt.barh(X.columns[sorted_feature_importance], 
#         model.feature_importances_[sorted_feature_importance], 
#         color='turquoise')
# plt.xlabel("CatBoost Feature Importance")
# plt.show()

# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_test)
# shap.summary_plot(shap_values, X_test, feature_names = X.columns[sorted_feature_importance])


# Now let's have a look what film it recommends

    # ALSO WANT TO SAVE THE MODEL SET UP - FEATURES AND HYPERPARAMETERS

    test_df = final_df[final_df.index.isin(df[df[user].isnull()].index)][model.feature_names_]

    out_df = pd.DataFrame({
        'tconst' : test_df.index, 
        user : model.predict(test_df)})

    out_df = out_df.set_index('tconst')

    pred_df = pred_df.merge(out_df, how='left', left_index=True, right_index=True)

    model_dict[user] = model
    feature_dict[user] = model.feature_names_

pred_df.to_csv('C:/Users/lwals/OneDrive/Desktop/CodingProjects/MovieRecommender/pred_scores_20231030.csv')

pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in feature_dict.items() ])).to_csv('C:/Users/lwals/OneDrive/Desktop/CodingProjects/MovieRecommender/model_features_20231030.csv')

# %%

def model_build(in_file: str, out_file: str, features_out: str, model_out = False):
    '''
    This file will build models for each of the users.
    
    IMPORTS

    INPUTS
    - in_file: This is the location of the ratings csv. Stored on the website 
               and the structure shouldn't change other than the adding of 
               users -> 'tconst', 'averageRating', 'numVotes', 'titleType', 
               'primaryTitle', 'startYear', 'runtimeMinutes', 'genre1', 
               'genre2', 'genre3', 'NoUserInput', ALL USER COLUMNS (where 
               the values are actual ratings)
    
    OUTPUT
    - out_file: This is where the output file will be saved to. Will be 
                'tconst', ALL USER COLUMNS (where the values are modeled 
                ratings)'''
    
    # Read in the file
    ratings_df = pd.read_csv(in_file)

    # Set the index to the tconst (Movie ID)
    ratings_df = ratings_df.set_index('tconst')

    # Imputing the missing ratings by iterative imputation
    imp_df = infill_missing_ratings(
        ratings_df, 
        list(ratings_df.iloc[:,10:].columns))

    # Encode the columns so the DataFrame is ready for modelling
    model_df = encode_and_combine_columns(imp_df, ['genre1', 'genre2','genre3'])

    # Rename a couple of the pesky columns that have dashes in them - Maybe 
    # look to generalise this a bit in case new genres are introduced.
    model_df.rename(
        {'Film-Noir': 'film_noir', 'Sci-Fi': 'sci_fi'}, 
        axis=1, inplace=True)

    # Remove no user input column, not necessary for modelling
    model_df.drop(
        ['NoUserInput', 'titleType', 'primaryTitle'], 
        axis=1, 
        inplace=True)

    # The base of the final df - all films rated by any user
    out_df = pd.DataFrame(
        index=ratings_df[ratings_df['NoUserInput'] == False].index)

    # Storing the users model to be saved later
    model_dict = {}

    # Storing the features used in the model to be saved later
    feature_dict = {}

    # For each user
    for user in model_df.columns[:-24]:

        # Creating the user's dataframe - only films they have seen
        user_df = model_df[model_df.index.isin(
            ratings_df[ratings_df[user].notnull()].index)]

        # Specifying target and features
        y = user_df[user]
        X = user_df.drop([user], axis=1)

        # Not necessary but could be used for future testing - EG "elbow" 
        # selection
        # res_dict = {}

        # Setting the min RMSE to be an arbitrarily high number
        min_rmse = 100

        # Cycling through 5 to 14 variables
        # NOTE: CHANGE NUM FEATURES TO BE DICTATED BY NUMBER OF FILMS THEY 
        # HAVE RATED
        # Look into literature around how many variables should be used based 
        # on how many values you have
        # Also, only model for people who have rated > 20 films (or whatever 
        # seems like enough), otherwise offer them the highest rated films on 
        # the database in order... 
        # NOTE: MAKE IT LESS RELIANT ON IMPUTED VALUES - ONLY USE COLUMNS THAT 
        # HAVE A CERTAIN NUMBER OF VALUES - or even any values at all - at the 
        # moment you could technically have rated no relevant films and still 
        # be part of the model
        for i in range(5, 15):
            
            # Specifying target and features
            y = user_df[user]
            X = user_df.drop([user], axis=1)

            # Splitting into test and train
            X_train, X_test, y_train, y_test = train_test_split(
                X, 
                y, 
                test_size = 0.2, 
                random_state=8)

            # Not exactly sure what pooling them does but I think it is used 
            # for CB to read in the variables
            train_dataset = cb.Pool(X_train, y_train) 
            test_dataset = cb.Pool(X_test, y_test)

            # The naive model
            model = cb.CatBoostRegressor(
                loss_function='RMSE',
                iterations=500,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=0.5)

            # Run it through feature selection - selecting i (5 to 14) features
            sf = model.select_features(
                train_dataset,
                features_for_select=X.columns, 
                num_features_to_select=i,
                verbose=100)

            # Take the selected features
            X = user_df[sf['selected_features_names']]

            # Train test split again
            X_train, X_test, y_train, y_test = train_test_split(
                X, 
                y, 
                test_size = 0.2, 
                random_state=8)

            # Pool again
            train_dataset = cb.Pool(X_train, y_train) 
            test_dataset = cb.Pool(X_test, y_test)

            # Creating the model
            model = cb.CatBoostRegressor(
                loss_function='RMSE',
                iterations=500,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=0.5)

            # Don't think it is nevessary to do hyper-parameter tuning just to 
            # decide how many features to use...
            # Building the grid of hyper-parameters
            # grid = {
            #     'iterations': [100, 150, 200],
            #     'learning_rate': [0.03, 0.1],
            #     'depth': [2, 4, 6, 8],
            #     'l2_leaf_reg': [0.2, 0.5, 1, 3]}
            
            # Grid search all the parameters
            # model.grid_search(grid, train_dataset, verbose=100)


            # Assess the models performance - What does the model predict in 
            # the test films? How does that compare to the actual? RMSE and R2
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            rmse = (np.sqrt(mean_squared_error(y_test, pred)))
            # r2 = r2_score(y_test, pred)

            # Want the number of features that minimises RMSE (NOTE: possibly 
            # change this to maximise R2 given the loss function used is RMSE)
            if rmse < min_rmse:
                chosen_num_features = i
                min_rmse = rmse

            # Not necessary but could be used for future testing - EG "elbow" 
            # selection
            # res_dict[i] = (rmse, r2)


        # Final run of the model
        # Define target and features
        y = user_df[user]
        X = user_df.drop([user], axis=1)

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size = 0.2, 
            random_state=8)

        # Pool again
        train_dataset = cb.Pool(X_train, y_train) 
        test_dataset = cb.Pool(X_test, y_test)

        # The naive mdel
        model = cb.CatBoostRegressor(
            loss_function='RMSE',
            iterations=500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=0.5)

        # Select (the chosen number of) features 
        sf = model.select_features(
            train_dataset,
            features_for_select=X.columns, 
            num_features_to_select=chosen_num_features,
            verbose=100)

        # Define the predictor variables (the ones you just selected)
        X = user_df[sf['selected_features_names']]

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size = 0.2, 
            random_state=8)

        # Pool it up
        train_dataset = cb.Pool(X_train, y_train) 
        test_dataset = cb.Pool(X_test, y_test)


        # NOTE: Wouldn't mind adding in an early stopping part to shorten this 
        # but doesn't look like grid search has it...
        model = cb.CatBoostRegressor(loss_function='RMSE')

        # NOTE: Should put a bit more thought into the relevance of all of 
        # these and the values selected...
        # NOTE: Should it go through another round of feature selection etc? 
        # Just to make sure I'm getting the "Best" model...
        # Define the hyperparameter grid to be searched.
        grid = {
            'iterations': [200, 500, 1000],
            'learning_rate': [0.03, 0.1],
            'depth': [2, 4, 6, 8],
            'l2_leaf_reg': [0.2, 0.5, 1, 3]}

        # Undertake the grid search
        model.grid_search(grid, train_dataset, verbose=100)

        # Final model has been defined, now we just run the predictions on the 
        # films the user hasn't seen
        to_pred_df = model_df[model_df.index.isin(
            ratings_df[ratings_df[user].isnull()].index)]

        # Running the predictions
        pred_df = pd.DataFrame({
            'tconst' : to_pred_df.index, 
            user : model.predict(to_pred_df[model.feature_names_])})

        # Set the index to the Film ID
        pred_df = pred_df.set_index('tconst')

        # Merge into the full table of predictions
        out_df = out_df.merge(
            pred_df, 
            how='left', 
            left_index=True, 
            right_index=True)

        # Possibly want to save the model.
        model_dict[user] = model

        # Want to Save the feature names
        feature_dict[user] = model.feature_names_

        break

    # pred_df.to_csv(out_file)

    # pd.DataFrame(dict([(k,pd.Series(v)) for k,v in feature_dict.items()])).to_csv(features_out)

    return out_df

out_df = model_build(
    'C:/Users/lwals/OneDrive/Desktop/CodingProjects/MovieRecommender/' + 
    'movieRatingsList_20230616.csv',
    'C:/Users/lwals/OneDrive/Desktop/CodingProjects/MovieRecommender/' + 
    'pred_scores_20231104.csv',
    'C:/Users/lwals/OneDrive/Desktop/CodingProjects/MovieRecommender/' + 
    'model_features_20231104.csv')