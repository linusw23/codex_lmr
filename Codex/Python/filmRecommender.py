import math
import numpy as np
import pandas as pd

YESNO = ["y", "n", "B2M"]

def similarity(impMovieRatingsList, movieNum, impMovieRatingsList2, movieNum2):
    '''Based on the given list of user movie ratings of movie, looks at how similar each users ratings is to movie2.
    The closer the ratings are, the higher the similarity score.'''

    d = 0

    for i in range(len(impMovieRatingsList.columns)):

        # print(i)
        # print((float(impMovieRatingsList.iloc[movieNum, i]))) # Check
        # print(float(impMovieRatingsList2.iloc[movieNum2, i])) # Check

        d += ((float(impMovieRatingsList.iloc[movieNum, i]) - float(impMovieRatingsList2.iloc[movieNum2, i]))**2)

#     print(d) # Check

    return(1/(1+math.sqrt(d)))

def recFilm(movieRatingsList, user):
    '''Provides a sorted list of the highest predicted ratings for films for a given user by comparing
    other user ratings for films the user hasn't seen to those same users ratings on films the user has seen
    and providing a similarity score. This score is then used to calculate a predicted score for each of the
    films the user hasn't seen, which is then sorted and returned.'''

    # calcCount = 0

    # Drop films that no user has rated.
    cols = list(movieRatingsList.columns[11:])
    impMovieRatingsList = movieRatingsList.dropna(axis=0, how='all', subset = cols)#.reset_index(drop = True)


    # Drop all irrelevant columns - leaving only film name, IMDB rating and user ratings
    impMovieRatingsList = impMovieRatingsList.drop(["tconst", "numVotes", "titleType", "startYear","runtimeMinutes", "genre1", "primaryTitle", "genre2", "genre3", "NoUserInput"], 1)
    # print(impMovieRatingsList)
    # Create a new dataframe with only the films that the user has rated
    impMovieRatingsList2 = impMovieRatingsList.dropna(subset=[user])
    # print(impMovieRatingsList2)
    # Want a df with just ratings
    impMovieRatingsList3 = impMovieRatingsList
    # print(impMovieRatingsList3)
    # Cut down the original dataframe to only films the user hasnt rated
    impMovieRatingsList = impMovieRatingsList[impMovieRatingsList[user].isna()]
    # print(impMovieRatingsList)
    # Create a list that is a mean of all the films' ratings
    m = impMovieRatingsList3.mean(axis=1)
    # print(m)

    # Imputation - for each user
    for col in range(len(impMovieRatingsList3.columns)):
        ratingCount = 0
        diffSum = 0

        # For each film that has a rating add the difference between the mean rating of that film and the user's rating of that film.
        i = 0
        for row in impMovieRatingsList3.index:
            # calcCount += 1
            if not math.isnan(impMovieRatingsList3.iloc[i,col]):
                ratingCount += 1
                diffSum += impMovieRatingsList3.iloc[i,col] - m[row]
            i += 1

        # Calculate the average difference between the user's rating and the average rating.
        finalDiff = diffSum/ratingCount
        # print(finalDiff)

        # For each film that the user has not rated, replace the NA value with
        # the mean of that film plus average difference between the user's
        # rating and the average rating.
        i = 0
        for row in impMovieRatingsList3.index:
            # calcCount += 1
            if math.isnan(impMovieRatingsList3.iloc[i,col]):
                impMovieRatingsList3.iloc[i,col] = m[row] + finalDiff
            i += 1

    # Assign the imputed columns to the imputed dataframes.
    for col in impMovieRatingsList3.columns:
        # calcCount += 1
        impMovieRatingsList[col] = impMovieRatingsList3[col]
        impMovieRatingsList2[col] = impMovieRatingsList3[col]


    # Drop the user from both dataframes
    impMovieRatingsList = impMovieRatingsList.drop(user, 1)
    impMovieRatingsList2 = impMovieRatingsList2.drop(user, 1)


    # print(impMovieRatingsList) # Check
    # print(impMovieRatingsList2) # Check

    finalRatings = []
    impMovieCount = 0

    # For each movie that the user hasn't rated...
    for movieNum in impMovieRatingsList.index:
        simList = []
        impMovieCount2 = 0

        # For each movie that the user has rated...
        for movieNum2 in impMovieRatingsList2.index:
            # calcCount += 1

            # If the movies aren't the same...which they shouldn't be
            if movieNum != movieNum2:

                # Create a tuple of how similar the film they haven't seen is with the film they have seen based on all users ratings.
                myTup = (str(similarity(impMovieRatingsList, impMovieCount, impMovieRatingsList2, impMovieCount2)),str(movieNum), str(movieNum2))
                simList.append(myTup)

            # Do this for all films the user has rated.
            impMovieCount2 += 1
        # print(simList)
        # Pick the top 5 films the user has seen that are most similar (according to all other user ratings) to the film the user hasn't seen
        top5 = sorted(simList, reverse = True)[:5]

        # print(top5) # Check 1

        numerator = 0
        denominator = 0

        # For each film in the top 5....
        for i in top5:
            # calcCount += 1

            # print(movieRatingsList.loc[int(i[1]), user]) # Check 2 - Should be NaNs
#             print(float(movieRatingsList.loc[int(i[2]), user])) # Check 1 - check against movieRatingsList

            # The numerator is the sum of each of the ratings the user has given for those films multiplied by the similarity score (for weighting)
            numerator += float(movieRatingsList.loc[int(i[2]), user]) * float(i[0])

            # The denominator is the sum of the top 5 similarity scores (for weighting)
            denominator += float(i[0])

        # The predicted rating for that film
        rating = numerator/denominator

#         print(rating) # Check 1

        # A tuple of the rating and the film's index.
        newTup = (rating, movieRatingsList.iloc[movieNum, 0])

#         print(newTup) # Check 1

        # Add both to a list of ratings, from which the recommendation will be picked
        finalRatings.append(newTup)

        impMovieCount += 1

#     print(finalRatings) # Check 3
    # print(calcCount)
    # print(len(finalRatings))
    # Return the sorted list of the highest predicted rated films for the user.
    return(sorted(finalRatings, reverse = True))

# def rateFilm(movieRatingsList, filmNumber, user):
#     '''This will allow the user to rate a given film and change their rating in the movieRatingsList'''

#     resp = None

#     # If the user has previously given the film a rating, this will tell them what they rated the film and asks them if they want to change it.
#     if not np.isnan(movieRatingsList.iloc[filmNumber, movieRatingsList.columns.get_loc(user)]):

#         # Ensures the user is answering "y", "n" or "B2M"
#         while resp not in YESNO:
#             resp = input("""You previously rated """ + str(movieRatingsList.iloc[filmNumber, 4]) + """ (""" + str(movieRatingsList.iloc[filmNumber, 5]) + """) """ + str(movieRatingsList.iloc[filmNumber, movieRatingsList.columns.get_loc(user)]) + """. Would you like to change your rating? (y/n)
# """)

#     # Raises the quit error, which needs to be dealt with when this function is called by taking the user back to the menu.
#     if resp == "B2M":
#         raise quit

#     # Keeps the user's rating the same.
#     if resp == "n":
#         print("Ok, your rating for " + str(movieRatingsList.iloc[filmNumber, 4]) + """ (""" + str(movieRatingsList.iloc[filmNumber, 5]) + """) """ + """is still """ + str(movieRatingsList.iloc[filmNumber, movieRatingsList.columns.get_loc(user)]) + """. You will now be directed back to the main menu.""")

#     # Asks them to give a new rating for the film.
#     else:

#         numberGiven = False
#         while not numberGiven:
#             try:
#                 rating = input("""What did you think of """ + str(movieRatingsList.iloc[filmNumber, 4]) + """ (""" + str(movieRatingsList.iloc[filmNumber, 5]) + """) """ + """? Please give a rating between 1 and 10.
# """)

#                 # Raises the quit error, which needs to be dealt with when this function is called by taking the user back to the menu.
#                 if rating == "B2M":
#                     raise quit

#                 # Ensures that the user is entering a float that is between 0 and 10 - if not tells them to enter it again.
#                 ratingFloat = float(rating)
#                 if ratingFloat >= 0.0 and ratingFloat <= 10.0:
#                     movieRatingsList.iloc[filmNumber, movieRatingsList.columns.get_loc(user)] = ratingFloat
#                     print("Ok, you have rated " + str(movieRatingsList.iloc[filmNumber, 4]) + """ (""" + str(movieRatingsList.iloc[filmNumber, 5]) + """) """ + str(movieRatingsList.iloc[filmNumber, movieRatingsList.columns.get_loc(user)]) + """.""")
#                     numberGiven = True
#                 else:
#                     print("""Your rating must be between 0 and 10. Try again.""")

#                 # Saves the user's rating to the database.
#                 movieRatingsList.to_csv('movieRatingsList.csv', index = False)

#             except ValueError:
#                 print("""That wasnt a number. Try again.
#                 """)


import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import json
import os
from functools import lru_cache
from urllib.parse import quote

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_TIMEOUT = float(os.getenv("TMDB_TIMEOUT", "6"))


def _request_json(url):
    resp = requests.get(url, timeout=TMDB_TIMEOUT)
    return json.loads(resp.content.decode('utf-8'))


@lru_cache(maxsize=4096)
def _first_movie_result(film_name, film_year):
    film_search = quote(str(film_name))
    url = (
        "https://api.themoviedb.org/3/search/movie?api_key=" + TMDB_API_KEY
        + "&language=en-AU&query=" + film_search
        + "&page=1&include_adult=false&year=" + str(film_year)
    )
    data = _request_json(url)
    results = data.get('results', [])
    if not results:
        return None
    return results[0]


@lru_cache(maxsize=128)
def _provider_list(country):
    url = (
        "https://api.themoviedb.org/3/watch/providers/movie?api_key=" + TMDB_API_KEY
        + "&language=en-" + country + "&watch_region=" + country
    )
    data = _request_json(url)
    return data.get('results', [])

def filmStreams(filmName, filmYear, country):
    first = _first_movie_result(filmName, filmYear)
    if not first:
        return []
    movie_id = first['id']
    url = "https://api.themoviedb.org/3/movie/" + str(movie_id) + "/watch/providers?api_key=" + TMDB_API_KEY
    a = _request_json(url)
    streamList = []
    # if user == "Lily":
    #     print(a['results']['UK'])
    #     # if 'flatrate' not in a['results']['UK'].keys():
    #     #     return streamList
    #     # for item in a['results']['AU']['flatrate']:
    #     #     streamList.append(item['provider_name'])

    if (country not in a['results'].keys()) or ('flatrate' not in a['results'][country].keys()):
        return streamList
    for item in a['results'][country]['flatrate']:
        streamList.append(item['provider_name'])

    return streamList

def descFilm(filmName, filmYear):
    first = _first_movie_result(filmName, filmYear)
    if not first:
        return ''
    return first.get('overview', '')

def filmPhoto(filmName, filmYear):
    first = _first_movie_result(filmName, filmYear)
    if not first or not first.get('poster_path'):
        return ''
    return 'https://image.tmdb.org/t/p/w500' + first['poster_path']

def mostSimilar(movieRatingsList, user):

    # cols = list(movieRatingsList.columns[11:])
    # return cols
    # movieRatingsListNew = movieRatingsList.dropna(axis=0, how='all', subset = cols)#.reset_index(drop = True)
    movieRatingsListNew = movieRatingsList.loc[movieRatingsList["NoUserInput"] == False]

    # # Drop all irrelevant columns - leaving only film name, IMDB rating and user ratings
    userRatings = movieRatingsListNew[user]
    movieRatingsListNew = movieRatingsListNew.iloc[:,11:].drop([user], axis=1)
    diffDict = {}
    for u in movieRatingsListNew.columns:
        # row = 0
        filmCount = 0
        diffSum = 0
        for i in movieRatingsListNew.index:
            if not (pd.isna(movieRatingsListNew.loc[i,u]) or pd.isna(userRatings[i])):
                # print(type(float(userRatings[i] - movieRatingsListNew.loc[i,u])))
                diffSum += (float(userRatings[i] - movieRatingsListNew.loc[i,u]))**2
                filmCount += 1
        if filmCount == 0:
            diffDict[u] = 9999999
        else:
            diffDict[u] = diffSum/filmCount
    closeUser = min(diffDict, key=diffDict.get)
    return closeUser

def streamImagesLinks(streamer, country):

    if country == "AU":
        linkDict = {'Netflix': 'https://www.netflix.com/',
            'Fetch TV': 'https://www.fetchtv.com.au/',
            'Amazon Prime Video': 'https://www.primevideo.com/',
            'Disney Plus': 'https://www.disneyplus.com/en-au',
            'Apple iTunes': 'https://www.apple.com/au/itunes/',
            'Google Play Movies': 'https://play.google.com/store/movies?hl=en_AU&gl=US',
            'BINGE': 'https://binge.com.au/',
            'Foxtel Now': 'https://www.foxtel.com.au/now/index.html',
            'Mubi': 'https://mubi.com/', 'GuideDoc': 'https://guidedoc.tv/',
            'Paramount Plus': 'https://www.paramountplus.com/au/',
            'Paramount+ Amazon Channel': 'https://www.primevideo.com/',
            'Stan': 'https://www.stan.com.au/',
            'Netflix Kids': 'https://www.netflix.com/',
            'Argo': 'https://watchargo.com/',
            'DocPlay': 'https://www.docplay.com/',
            'Quickflix': '',
            'Acorn TV': 'https://au.acorn.tv/',
            'Quickflix Store': '',
            'Apple TV Plus': 'https://www.apple.com/au/apple-tv-plus/',
            'Amazon Video': 'https://www.primevideo.com/',
            'Sun Nxt': 'https://www.sunnxt.com/',
            'YouTube': 'https://www.youtube.com/',
            'Curiosity Stream': 'https://curiositystream.com/',
            'Spamflix': 'https://spamflix.com/home.do',
            'Funimation Now': 'https://www.funimation.com/',
            'DOCSVILLE': 'https://www.docsville.com/',
            'WOW Presents Plus': 'https://www.wowpresentsplus.com/',
            '7plus': 'https://7plus.com.au/',
            'Magellan TV': 'https://www.magellantv.com/',
            'ITV Hub':'https://www.itv.com/',
            'ABC iview': 'https://iview.abc.net.au/',
            'BroadwayHD': 'https://www.broadwayhd.com/',
            'SBS On Demand': 'https://www.sbs.com.au/ondemand/',
            'Filmzie': 'https://filmzie.com/',
            'Dekkoo': 'https://www.dekkoo.com/',
            'Beamafilm': 'https://beamafilm.com/landing-guest',
            'True Story': 'https://www.truestory.film/',
            'Classix': 'https://www.classixapp.com/',
            'Hoichoi': 'https://www.hoichoi.tv/',
            'BritBox': 'https://www.britbox.com/au/',
            'Eventive': 'https://eventive.org/',
            'YouTube Premium': 'https://www.youtube.com/premium',
            'Telstra TV': 'https://www.telstra.com.au/entertainment/tv-movies/telstra-tv',
            'YouTube Free': 'https://www.youtube.com/',
            'OzFlix': 'https://www.ozflix.tv/#!/',
            'Pantaflix': 'https://www.pantaflix.com/',
            'Tubi TV': 'https://tubitv.com/',
            'Kanopy': 'https://www.kanopy.com/',
            'Microsoft Store': 'https://www.microsoft.com/en-au/store/movies-and-tv',
            'Shudder': 'https://www.shudder.com/',
            'HiDive': 'https://www.hidive.com/',
            'Plex': 'https://www.plex.tv/',
            'Dogwoof On Demand': 'https://watch.dogwoof.com/',
            'Cultpix': 'https://www.cultpix.com/',
            'MUBI': 'https://mubi.com/',
            'FilmBox+': 'https://www.filmbox.com/int/home?locale=en-US',
            'Takflix': 'https://takflix.com/en',
            'Starz Play Amazon Channel': 'https://www.primevideo.com/',
            'Netflix basic with Ads': 'https://www.netflix.com/',
            'Shudder Amazon Channel': 'https://www.primevideo.com/'}

        for s in _provider_list("AU"):
            if s['provider_name'] == streamer and streamer in linkDict.keys():
                image_str = "https://image.tmdb.org/t/p/original" + s['logo_path']
                return (image_str, linkDict[streamer])

    elif country == "GB":
        linkDict = {'Netflix': 'https://www.netflix.com/',
            'Amazon Prime Video': 'https://www.primevideo.com/',
            'Paramount+ Amazon Channel': 'https://www.primevideo.com/',
            'BFI Player Amazon Channel': 'https://www.primevideo.com/',
            'Freevee Amazon Channel': 'https://www.primevideo.com/',
            'Hayu Amazon Channel': 'https://www.primevideo.com/',
            'ShortsTV Amazon Channel': 'https://www.primevideo.com/',
            'MGM Amazon Channel': 'https://www.primevideo.com/',
            'CuriosityStream Amazon Channel': 'https://www.primevideo.com/',
            'DocuBay Amazon Channel': 'https://www.primevideo.com/',
            'Full Moon Amazon Channel': 'https://www.primevideo.com/',
            'Pokémon Amazon Channel': 'https://www.primevideo.com/',
            'Shout! Factory Amazon Channel': 'https://www.primevideo.com/',
            'Eros Now Amazon Channel': 'https://www.primevideo.com/',
            'FilmBox Live Amazon Channel': 'https://www.primevideo.com/',
            'MotorTrend Amazon Channel': 'https://www.primevideo.com/',
            'Shudder Amazon Channel': 'https://www.primevideo.com/',
            'Mubi Amazon Channel': 'https://www.primevideo.com/',
            'AcornTV Amazon Channel': 'https://www.primevideo.com/',
            'BritBox Amazon Channel': 'https://www.primevideo.com/',
            'Fandor Amazon Channel': 'https://www.primevideo.com/',
            'Flix Premiere': 'https://flixpremiere.com/',
            'Revry': 'https://www.revry.tv/',
            'ARROW': 'https://www.arrow-player.com/browse',
            'W4free': 'https://apps.apple.com/gb/app/w4free/id1483415668',
            'Paus': 'https://paus.tv/',
            'Yupp TV': 'https://www.yupptv.com/',
            'My5': 'https://www.channel5.com/',
            'Curzon Home Cinema':'https://homecinema.curzon.com/',
            'Disney Plus': 'https://www.disneyplus.com/',
            'WOW Presents Plus': 'https://www.wowpresentsplus.com/',
            'Starz Play Amazon Channel': 'https://www.primevideo.com/',
            'Starz':'https://www.starz.com/',
            'Apple iTunes': 'https://www.apple.com/au/itunes/',
            'Google Play Movies': 'https://play.google.com/store/movies?hl=en_GB&gl=GB',
            'Sky Go': "https://www.sky.com/watch/sky-go/windows",
            'Sky Store': 'https://www.skystore.com/',
            'Now TV': 'https://www.nowtv.com/',
            'Now TV Cinema': 'https://www.nowtv.com/',
            'Virgin TV Go': 'https://virgintvgo.virginmedia.com/en.html',
            'BBC iPlayer': 'https://www.bbc.co.uk/iplayer',
            'Rakuten TV': 'https://rakuten.tv/uk/movies',
            'Mubi': 'https://mubi.com/',
            'GuideDoc': 'https://guidedoc.tv/',
            'Paramount Plus': 'https://www.paramountplus.com/au/',
            'Stan': 'https://www.stan.com.au/',
            'Netflix Kids': 'https://www.netflix.com/',
            'Argo': 'https://watchargo.com/',
            'DocPlay': 'https://www.docplay.com/',
            'Quickflix': '',
            'Acorn TV': 'https://acorn.tv/',
            'Quickflix Store': '',
            'Apple TV Plus': 'https://www.apple.com/uk/apple-tv-plus/',
            'STUDIOCANAL PRESENTS Apple TV Channel': 'https://www.apple.com/uk/apple-tv-plus/',
            'STV Player':'https://www.stv.tv/',
            'Amazon Video': 'https://www.primevideo.com/',
            'Arrow Video Amazon Channel': 'https://www.primevideo.com/',
            'Pluto TV':'https://pluto.tv/',
            'Sun Nxt': 'https://www.sunnxt.com/',
            'YouTube': 'https://www.youtube.com/',
            'Curiosity Stream': 'https://curiositystream.com/',
            'Spamflix': 'https://spamflix.com/home.do',
            'Funimation Now': 'https://www.funimation.com/',
            'DOCSVILLE': 'https://www.docsville.com/',
            'All 4': 'https://www.channel4.com/',
            'Curzon Home Cinema': 'https://homecinema.curzon.com/',
            'WOW Presents Plus': 'https://www.wowpresentsplus.com/',
            '7plus': 'https://7plus.com.au/',
            'Magellan TV': 'https://www.magellantv.com/',
            'BFI Player': 'https://player.bfi.org.uk/',
            'BroadwayHD': 'https://www.broadwayhd.com/',
            'Discovery+ Amazon Channel': 'https://www.amazon.co.uk/Discovery-Communications-discovery-Stream-Shows/dp/B08F8XQLQZ',
            'FilmBox+': 'https://www.filmbox.com/',
            'Filmzie': 'https://filmzie.com/',
            'Dekkoo': 'https://www.dekkoo.com/',
            'Beamafilm': 'https://beamafilm.com/landing-guest',
            'True Story': 'https://www.truestory.film/',
            'DocAlliance Films': 'https://dafilms.com/',
            'Classix': 'https://www.classixapp.com/',
            'Hoichoi': 'https://www.hoichoi.tv/',
            'Chili': 'https://uk.chili.com/',
            'BritBox': 'https://www.britbox.com/',
            'Discovery Plus': 'https://www.discoveryplus.com/',
            'Eventive': 'https://eventive.org/',
            'YouTube Premium': 'https://www.youtube.com/premium',
            'Telstra TV': 'https://www.telstra.com.au/entertainment/tv-movies/telstra-tv',
            'YouTube Free': 'https://www.youtube.com/',
            'OzFlix': 'https://www.ozflix.tv/#!/',
            'Pantaflix': 'https://www.pantaflix.com/',
            'Tubi TV': 'https://tubitv.com/',
            'Kanopy': 'https://www.kanopy.com/',
            'Microsoft Store': 'https://www.microsoft.com/en-gb/store/movies-and-tv',
            'Shudder': 'https://www.shudder.com/',
            'HiDive': 'https://www.hidive.com/',
            'Plex': 'https://www.plex.tv/',
            'Dogwoof On Demand': 'https://watch.dogwoof.com/',
            'Cultpix': 'https://www.cultpix.com/',
            'MUBI': 'https://mubi.com/',
            'ITV Hub': 'https://www.itv.com/hub/itv',
            'MUBI Amazon Channel': 'https://www.primevideo.com/',
            'Discovery Plus Amazon Channel': 'https://www.primevideo.com/',
            'Icon Film Amazon Channel': 'https://www.primevideo.com/',
            'Curzon Amazon Channel': 'https://www.primevideo.com/',
            'Hallmark TV Amazon Channel': 'https://www.primevideo.com/',
            'Studiocanal Presents Amazon Channel': 'https://www.primevideo.com/',
            'Sundance Now Amazon Channel': 'https://www.primevideo.com/',
            'Realeyz Amazon Channel': 'https://www.primevideo.com/',
            'Takflix': 'https://takflix.com/en',
            'Lionsgate Plus': 'https://www.lionsgate.com/',
            'Klassiki': 'https://films.klassiki.online/',
            'Viaplay': 'https://viaplay.com/',
            'Netflix basic with Ads': 'https://www.netflix.com/'}

        for s in _provider_list("GB"):
            if s['provider_name'] == streamer and streamer in linkDict.keys():
                image_str = "https://image.tmdb.org/t/p/original" + s['logo_path']
                return (image_str, linkDict[streamer])

    elif country == "SA":
        linkDict = {'Netflix': 'https://www.netflix.com/',
            'OSN': 'https://www.osn.com/en-sa/home',
            'STARZPLAY': 'https://starzplay.com/',
            'iWantTFC': 'https://www.iwanttfc.com/',
            'Shahid VIP': 'https://shahid.mbc.net/',
            'aha': 'https://www.aha.video/',
            'Paramount+ Amazon Channel': 'https://www.primevideo.com/',
            'Amazon Prime Video': 'https://www.primevideo.com/',
            'BFI Player Amazon Channel': 'https://www.primevideo.com/',
            'Freevee Amazon Channel': 'https://www.primevideo.com/',
            'Hayu Amazon Channel': 'https://www.primevideo.com/',
            'ShortsTV Amazon Channel': 'https://www.primevideo.com/',
            'MGM Amazon Channel': 'https://www.primevideo.com/',
            'CuriosityStream Amazon Channel': 'https://www.primevideo.com/',
            'DocuBay Amazon Channel': 'https://www.primevideo.com/',
            'Full Moon Amazon Channel': 'https://www.primevideo.com/',
            'Pokémon Amazon Channel': 'https://www.primevideo.com/',
            'Shout! Factory Amazon Channel': 'https://www.primevideo.com/',
            'Eros Now Amazon Channel': 'https://www.primevideo.com/',
            'FilmBox Live Amazon Channel': 'https://www.primevideo.com/',
            'MotorTrend Amazon Channel': 'https://www.primevideo.com/',
            'Shudder Amazon Channel': 'https://www.primevideo.com/',
            'Mubi Amazon Channel': 'https://www.primevideo.com/',
            'AcornTV Amazon Channel': 'https://www.primevideo.com/',
            'BritBox Amazon Channel': 'https://www.primevideo.com/',
            'Fandor Amazon Channel': 'https://www.primevideo.com/',
            'Flix Premiere': 'https://flixpremiere.com/',
            'Revry': 'https://www.revry.tv/',
            'ARROW': 'https://www.arrow-player.com/browse',
            'W4free': 'https://apps.apple.com/gb/app/w4free/id1483415668',
            'Paus': 'https://paus.tv/',
            'Yupp TV': 'https://www.yupptv.com/',
            'My5': 'https://www.channel5.com/',
            'Curzon Home Cinema':'https://homecinema.curzon.com/',
            'Disney Plus': 'https://www.disneyplus.com/',
            'WOW Presents Plus': 'https://www.wowpresentsplus.com/',
            'Starz Play Amazon Channel': 'https://www.primevideo.com/',
            'Starz':'https://www.starz.com/',
            'Apple iTunes': 'https://www.apple.com/SA/itunes/',
            'Zee5': 'https://www.zee5.com/',
            'FlixOlé': 'https://flixole.com/',
            'Google Play Movies': 'https://play.google.com/store/movies?hl=en_SA&gl=SA',
            'Sky Go': "https://www.sky.com/watch/sky-go/windows",
            'Sky Store': 'https://www.skystore.com/',
            'Now TV': 'https://www.nowtv.com/',
            'Now TV Cinema': 'https://www.nowtv.com/',
            'Virgin TV Go': 'https://virgintvgo.virginmedia.com/en.html',
            'BBC iPlayer': 'https://www.bbc.co.uk/iplayer',
            'Rakuten TV': 'https://rakuten.tv/uk/movies',
            'Mubi': 'https://mubi.com/',
            'GuideDoc': 'https://guidedoc.tv/',
            'Paramount Plus': 'https://www.paramountplus.com/au/',
            'Stan': 'https://www.stan.com.au/',
            'Netflix Kids': 'https://www.netflix.com/',
            'Argo': 'https://watchargo.com/',
            'DocPlay': 'https://www.docplay.com/',
            'Quickflix': '',
            'Acorn TV': 'https://acorn.tv/',
            'Quickflix Store': '',
            'VIX ': 'https://vix.com/',
            'Apple TV Plus': 'https://www.apple.com/uk/apple-tv-plus/',
            'STUDIOCANAL PRESENTS Apple TV Channel': 'https://www.apple.com/uk/apple-tv-plus/',
            'STV Player':'https://www.stv.tv/',
            'Amazon Video': 'https://www.primevideo.com/',
            'Arrow Video Amazon Channel': 'https://www.primevideo.com/',
            'Pluto TV':'https://pluto.tv/',
            'Sun Nxt': 'https://www.sunnxt.com/',
            'YouTube': 'https://www.youtube.com/',
            'Curiosity Stream': 'https://curiositystream.com/',
            'Spamflix': 'https://spamflix.com/home.do',
            'Spuul': 'https://www.spuul.com/',
            'Funimation Now': 'https://www.funimation.com/',
            'DOCSVILLE': 'https://www.docsville.com/',
            'All 4': 'https://www.channel4.com/',
            'Curzon Home Cinema': 'https://homecinema.curzon.com/',
            'WOW Presents Plus': 'https://www.wowpresentsplus.com/',
            '7plus': 'https://7plus.com.au/',
            'Magellan TV': 'https://www.magellantv.com/',
            'BFI Player': 'https://player.bfi.org.uk/',
            'BroadwayHD': 'https://www.broadwayhd.com/',
            'Discovery+ Amazon Channel': 'https://www.amazon.co.uk/Discovery-Communications-discovery-Stream-Shows/dp/B08F8XQLQZ',
            'FilmBox+': 'https://www.filmbox.com/',
            'Filmzie': 'https://filmzie.com/',
            'Dekkoo': 'https://www.dekkoo.com/',
            'Beamafilm': 'https://beamafilm.com/landing-guest',
            'True Story': 'https://www.truestory.film/',
            'DocAlliance Films': 'https://dafilms.com/',
            'Classix': 'https://www.classixapp.com/',
            'Public Domain Movies': 'http://publicdomainmovies.net/',
            'Hoichoi': 'https://www.hoichoi.tv/',
            'Rakuten Viki': 'https://www.viki.com/',
            'iQIYI': 'https://www.iq.com/?lang=en_us',
            'Chili': 'https://uk.chili.com/',
            'BritBox': 'https://www.britbox.com/',
            'Discovery Plus': 'https://www.discoveryplus.com/',
            'Eventive': 'https://eventive.org/',
            'CONtv': 'https://www.contv.com/',
            'YouTube Premium': 'https://www.youtube.com/premium',
            'Telstra TV': 'https://www.telstra.com.au/entertainment/tv-movies/telstra-tv',
            'YouTube Free': 'https://www.youtube.com/',
            'OzFlix': 'https://www.ozflix.tv/#!/',
            'Pantaflix': 'https://www.pantaflix.com/',
            'Tubi TV': 'https://tubitv.com/',
            'Kanopy': 'https://www.kanopy.com/',
            'Microsoft Store': 'https://www.microsoft.com/en-gb/store/movies-and-tv',
            'Shudder': 'https://www.shudder.com/',
            'HiDive': 'https://www.hidive.com/',
            'Plex': 'https://www.plex.tv/',
            'Dogwoof On Demand': 'https://watch.dogwoof.com/',
            'Cultpix': 'https://www.cultpix.com/',
            'MUBI': 'https://mubi.com/',
            'TOD': 'https://www.tod.tv/en/',
            'Takflix': 'https://takflix.com/en'}

        for s in _provider_list("SA"):
            if s['provider_name'] == streamer and streamer in linkDict.keys():
                image_str = "https://image.tmdb.org/t/p/original" + s['logo_path']
                return (image_str, linkDict[streamer])


    return None


    # return a['results'][0]


    # titles = soup.find_all('h2')
    # dates = soup.find_all('span', {"class": "release_date"})
    # URLs = soup.find_all('a', {"class": "result"}, href=True)

    # filmURL = None
    # for i in range(len(titles)):
    #     if titles[i].contents[0] == filmName:
    #         if int(dates[i].contents[0][-4:]) == filmYear:
    #             filmURL = "https://www.themoviedb.org" + URLs[0*2]['href'][:-15] + "/watch?language=en-AU"

    # if filmURL:
    #     streamList = []
    #     page = requests.get(filmURL, headers=headers)
    #     soup = BeautifulSoup(page.content, 'html.parser')
    #     streams = soup.find_all('a', {'rel':'noopener'}, title=True)

    #     for i in range(len(streams)):
    #         phraseLen = 10 + len(filmName)
    #         if streams[i]['title'][:5] == "Watch" and (streams[i]['title'][phraseLen:] not in streamList):
    #             streamList.append(streams[i]['title'][phraseLen:])

    #     streamString = ""
    #     for stream in streamList:
    #         streamString = streamString + stream + ", "

    # return(streamString)
        # print(filmName + " can be found on " + streamString[:-2] + ".")
        # print("Unfortunately I was unable to locate any streaming service showing " + filmName + ". Hopefully you can find it!")

#         print(title.contents[0])


# # Check
# movieRatingsListTest = pd.read_csv("/home/lwalsh23/movieRatingsList.csv")
# print(recFilm(movieRatingsListTest,"Felix"))

# Check


# Check
# print(filmStreams("Braveheart", 1995, 'GB'))

# Check
# print(descFilm("Braveheart", 1995))

# Check
# movieRatingsList = pd.read_csv("/home/lwalsh23/movieRatingsList.csv")
# # print(mostSimilar(movieRatingsList, "Lily"))
# # print(recFilm(movieRatingsList, "Sam"))
# print(recFilm(movieRatingsList, 'Hannah'))

# Check
# print(filmPhoto("Braveheart", 1995))
# print("----------------------")
# print(movieRatingsList.iloc[:,11:].count().sort_values(ascending=False)[:5].index[0])

# print(streamImagesLinks("Netflix"))

# accountDetails = pd.read_csv("/home/lwalsh23/accountDetails.csv")

# print(accountDetails.loc[accountDetails['User'] == 'Linus', 'Password'].values[0])
# print(accountDetails['User'].eq('Linus').any())


