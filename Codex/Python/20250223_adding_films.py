# First need to get the current data from website... Manually download from PythonAnywhere

import pandas as pd

movieRatingsList = pd.read_csv("movieRatingsList.csv")

movieRatingsList

# Getting the IMDB dataset - downloading and converting to csv.

# Libraries
import re
from urllib import request
import gzip
import shutil
import pandas as pd
from multiprocessing.pool import ThreadPool

def download_url(url):
    # Download process
    print("downloading: ",url)
    file_title = re.split(pattern='/', string=url)[-1]
    urlrtv = request.urlretrieve(url=url, filename=file_title)
    
    # for ".tsv" to ".csv"
    title = re.split(pattern=r'\.tsv', string=file_title)[0] +".csv"
    
    # Unzip ".gz" file
    with gzip.open(file_title, 'rb') as f_in:
        with open(title, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            
urls = ["https://datasets.imdbws.com/title.basics.tsv.gz"
        ,"https://datasets.imdbws.com/title.ratings.tsv.gz"
          ]

results = ThreadPool(5).imap_unordered(download_url, urls)
for r in results:
    print(r)

    # Read into python
import numpy as np

title_basics_data = pd.read_csv("title.basics.csv",sep="\\t",engine = "python",na_values=["\\N"])
title_basics = pd.DataFrame(title_basics_data, columns= ['tconst','titleType','primaryTitle','startYear','runtimeMinutes','genres','isAdult'])
title_basics = title_basics.loc[(title_basics['isAdult'] == 0) 
                                & (title_basics['titleType'].isin(['movie','tvMovie'])) 
                                & (title_basics['genres'] != "NaN")]
title_basics = title_basics[['tconst','titleType','primaryTitle','startYear','runtimeMinutes','genres']]
title_basics = title_basics.dropna().reset_index(drop=True)


title_ratings_data = pd.read_csv("title.ratings.csv",sep="\\t",engine = "python",na_values=["\\N"])
title_ratings = pd.DataFrame(title_ratings_data, columns= ['tconst','averageRating','numVotes'])
title_ratings = title_ratings.loc[(title_ratings['numVotes'] > 999)]
title_ratings = title_ratings.dropna().reset_index(drop=True)


new_df = title_basics.merge(title_ratings, on='tconst', how='left')
new_df = new_df.dropna().reset_index(drop=True)
genres = new_df['genres'].str.split(',', expand=True)
new_df['genre1'] = genres[0]
new_df['genre2'] = genres[1]
new_df['genre3'] = genres[2]
new_df = new_df[['tconst','averageRating','numVotes','titleType','primaryTitle','startYear','runtimeMinutes','genre1','genre2','genre3']]

# Merging in everyone's ratings
ratings_cols = list(movieRatingsList.columns[10:])
ratings_cols.append('tconst')
ratings = movieRatingsList[ratings_cols]

update = new_df.merge(ratings, how='left', on='tconst')
update['NoUserInput'] = update['NoUserInput'].fillna(True)
update['numVotes']=update['numVotes'].astype(int) 
update['startYear']=update['startYear'].astype(int)
update['runtimeMinutes']=update['runtimeMinutes'].astype(int) 

update = update.sort_values(by="numVotes", ascending=False)
# update.to_csv('test.csv',index=False)
update.iloc[:,:-1].to_csv('movieRatingsList.csv',index=False)

# accountDetails = pd.read_csv("C:/Users/lwals/OneDrive/Desktop/CodingProjects/MovieRecommender/accountDetails.csv")
# accountDetails[]