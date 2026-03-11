from math import pi
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_genre_ratings_df(df, user_list, spider=True):
    out_df = pd.DataFrame()
    user_start = df.columns.get_loc('NoUserInput') + 1
    numeric_ratings = df.iloc[:, user_start:].apply(pd.to_numeric, errors='coerce')

    if spider:
        genre_list = ['Comedy', 'Drama', 'Action', 'Crime', 'Adventure', 'Horror']
    else:
        genre_list = list(set(list(df['genre1'].unique()) + list(df['genre2'].unique()) + list(df['genre3'].unique())))
        if np.nan in genre_list:
            genre_list.remove(np.nan)

    out_df['user'] = ['average']
    for genre in genre_list:
        genre_mask = (
            ((df['genre1'] == genre) | (df['genre2'] == genre) | (df['genre3'] == genre)) &
            (df['NoUserInput'] == False)
        )
        arr = numeric_ratings.loc[genre_mask].to_numpy()
        out_df[genre] = [np.nanmean(arr) if arr.size else 0]

    for user in user_list:
        user_scores = [user]
        for genre in genre_list:
            user_scores.append(
                pd.to_numeric(df[
                    ((df['genre1'] == genre) | (df['genre2'] == genre) | (df['genre3'] == genre)) &
                    (df['NoUserInput'] == False)
                ][user], errors='coerce').mean()
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
    output_dir = Path(save_loc)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_dir / 'spider_chart.png'), dpi=180, bbox_inches='tight')
    plt.close()


def top_5_genres(df, user):
    genre_ratings = build_genre_ratings_df(df, [user], False)
    genre_ratings.set_index('user', inplace=True)
    return genre_ratings.loc[user].sort_values(ascending=False)[:5]


def most_sim_user(df, user):
    user_start = df.columns.get_loc('NoUserInput') + 1
    user_df = df[df[user].notnull()].iloc[:, user_start:].copy()
    user_df = user_df.apply(pd.to_numeric, errors="coerce")

    other_users = list(user_df.columns)
    other_users.remove(user)

    min_dist = 10
    closest = None

    for other in other_users:
        shared_df = user_df[user_df[other].notnull()][[user, other]].dropna().copy()
        if len(shared_df) > 0:
            shared_df['diff_sq'] = (shared_df[user] - shared_df[other]) ** 2
            msd = shared_df['diff_sq'].mean()
            if msd < min_dist:
                min_dist = msd
                closest = other

    return closest if closest else "No close match yet"
