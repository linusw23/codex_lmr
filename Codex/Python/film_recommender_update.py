from math import pi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_genre_ratings_df(df, user_list, spider=True):
    out_df = pd.DataFrame()

    if spider:
        genre_list = ['Comedy', 'Drama', 'Action', 'Crime', 'Adventure', 'Horror']
    else:
        genre_list = list(set(list(df['genre1'].unique()) + list(df['genre2'].unique()) + list(df['genre3'].unique())))
        if np.nan in genre_list:
            genre_list.remove(np.nan)

    out_df['user'] = ['average']
    for genre in genre_list:
        out_df[genre] = [np.nanmean(df[
            ((df['genre1'] == genre) | (df['genre2'] == genre) | (df['genre3'] == genre)) &
            (df['NoUserInput'] == False)].iloc[:, 8:])]

    for user in user_list:
        user_scores = [user]
        for genre in genre_list:
            user_scores.append(
                df[
                    ((df['genre1'] == genre) | (df['genre2'] == genre) | (df['genre3'] == genre)) &
                    (df['NoUserInput'] == False)
                ][user].mean()
            )
        out_df.loc[len(out_df)] = user_scores

    return out_df.fillna(0)


def genre_spider_chart(data, save_loc):
    categories = list(data)[1:]
    count = len(categories)

    angles = [n / float(count) * 2 * pi for n in range(count)]
    angles += angles[:1]

    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, fontsize=12)
    ax.xaxis.set_tick_params(pad=10)
    ax.set_rlabel_position(0)
    plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", '8', '10'], color="grey", size=12)
    plt.ylim(0, 10)

    values = data.loc[1].drop('user').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=3, linestyle='solid', label=data.loc[1, 'user'], color='darkred')

    values = data.loc[0].drop('user').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label="LMR Average", color='darkgrey')

    if data.shape[0] == 3:
        values = data.loc[2].drop('user').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=data.loc[2, 'user'], color='#607D8B')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=data.shape[0], fontsize=14)
    plt.savefig(save_loc + '/spider_chart.png', dpi=500, bbox_inches='tight')
    plt.close()


def top_5_genres(df, user):
    genre_ratings = build_genre_ratings_df(df, [user], False)
    genre_ratings.set_index('user', inplace=True)
    return genre_ratings.loc[user].sort_values(ascending=False)[:5]


def most_sim_user(df, user):
    user_df = df[df[user].notnull()].iloc[:, 8:]
    other_users = list(user_df.columns)
    other_users.remove(user)

    min_dist = 10
    closest = None

    for other in other_users:
        shared_df = user_df[user_df[other].notnull()][[user, other]]
        if len(shared_df) > 0:
            shared_df['diff_sq'] = shared_df.apply(lambda row: (row[user] - row[other]) ** 2, axis=1)
            msd = shared_df['diff_sq'].mean()
            if msd < min_dist:
                min_dist = msd
                closest = other

    return closest
