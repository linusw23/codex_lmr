from flask import Flask, request, session, redirect, send_from_directory
from pathlib import Path
import os
import shutil
from db_storage import install_bootstrap, read_table_for_csv, write_table_for_csv, database_ready
import pandas as pd
import numpy as np
from fuzzywuzzy import process
from filmRecommender import filmStreams, descFilm, filmPhoto, streamImagesLinks
from film_recommender_update import build_genre_ratings_df, genre_spider_chart, top_5_genres, most_sim_user
import random
import datetime
import io
import zipfile
import re



# Setting up app paths and runtime config.
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BASE_DIR / 'Other Files'
DATA_DIR = Path(os.getenv('DATA_DIR', str(DEFAULT_DATA_DIR)))
HTML_DIR = BASE_DIR / 'HTML'
ASSETS_DIR = BASE_DIR / 'assets'

if DATA_DIR != DEFAULT_DATA_DIR and not (DATA_DIR / 'movieRatingsList.csv').exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for fname in ['movieRatingsList.csv', 'pred_scores.csv', 'fp_pred_scores.csv', 'accountDetails.csv']:
        src = DEFAULT_DATA_DIR / fname
        dst = DATA_DIR / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-only-change-me')

# Route legacy CSV access to the database layer.
USE_DATABASE = os.getenv("USE_DATABASE", "1") == "1"
AUTO_BOOTSTRAP_DB = os.getenv("AUTO_BOOTSTRAP_DB", "0") == "1"
if USE_DATABASE:
    try:
        install_bootstrap(auto_bootstrap=AUTO_BOOTSTRAP_DB)
        if not database_ready():
            print("DB not ready; falling back to CSV mode for this run.")
            USE_DATABASE = False
    except Exception as ex:
        print(f"DB init failed ({ex}); falling back to CSV mode for this run.")
        USE_DATABASE = False

if USE_DATABASE:
    _pd_read_csv = pd.read_csv
    _pd_to_csv = pd.DataFrame.to_csv
    _mapped_csv = {"accountDetails.csv", "movieRatingsList.csv", "pred_scores.csv", "fp_pred_scores.csv"}

    def _csv_name(path_or_buf):
        if isinstance(path_or_buf, Path):
            return path_or_buf.name
        if isinstance(path_or_buf, str):
            return Path(path_or_buf).name
        return None

    def _db_read_csv(path_or_buf, *args, **kwargs):
        name = _csv_name(path_or_buf)
        if name in _mapped_csv:
            return read_table_for_csv(name, index_col=kwargs.get("index_col"))
        return _pd_read_csv(path_or_buf, *args, **kwargs)

    def _db_to_csv(self, path_or_buf=None, *args, **kwargs):
        name = _csv_name(path_or_buf)
        if name in _mapped_csv:
            write_table_for_csv(name, self.copy(), index=kwargs.get("index", True))
            return None
        return _pd_to_csv(self, path_or_buf, *args, **kwargs)

    pd.read_csv = _db_read_csv
    pd.DataFrame.to_csv = _db_to_csv


@app.route('/assets/<path:filename>')
def assets(filename):
    return send_from_directory(ASSETS_DIR, filename)

@app.route('/', methods=["GET", "POST"])
def home():

    '''The home page the user is greeted by before they log in'''

    session['logged_in'] = False

    if 'recCount' not in session:
        session['recCount'] = 0

    # Setting up the movieRatingsList - reading from the csv saved on
    # PythonAnywhere
    movieRatingsList = pd.read_csv("movieRatingsList.csv")

    # Getting the average rating of LMR users to decide what film should be
    # recommended to non-logged in users
    movieRatingsList['AverageUserRating'] = movieRatingsList.iloc[:,11:].mean(
        axis=1)
    movieRatingsList.sort_values(
            by=['AverageUserRating', 'numVotes'],
            inplace=True,
            ascending=False)
    movieRatingsList.reset_index(
                inplace=True,
                drop=True)

    # Getting info about the current film to display
    film_tconst = str(movieRatingsList.loc[session['recCount'],'tconst'])
    film_name = str(movieRatingsList.loc[session['recCount'],'primaryTitle'])
    film_year = movieRatingsList.loc[session['recCount'],'startYear']
    film_genres = movieRatingsList.loc[
        session['recCount'],
        ['genre1', 'genre2', 'genre3']].str.cat(sep=' • ')

    try: film_desc = descFilm(str(film_name), int(film_year))
    except: film_desc = ''
    try: film_photo = filmPhoto(film_name, film_year)
    except: film_photo = ''

    # Geting streams
    try: film_streams = filmStreams(film_name, film_year, 'AU')
    except: film_streams = []

    stream_string = ''
    for stream in film_streams:
        imageLink = streamImagesLinks(stream, 'AU')
        if imageLink:
            stream_string += ''' <a href="{link}"><img src="{photo}" style="max-width: 40px; height: auto; margin-bottom: 4px;" alt="{stream}"></a>'''.format(photo=imageLink[0], stream=stream, link=imageLink[1])

    today = datetime.date.today().strftime("%A")

    # Hall of Fame
    hall_of_fame = movieRatingsList.iloc[:,11:-1].count().sort_values(
        ascending=False)[:5]


    if request.method == "POST":

        # If the user clicks on the film to get more detail, send them to the
        # film details page
        tconst = request.form.get("tconst")
        if tconst:
            session['tconst'] = tconst
            session.modified = True
            return redirect('/filmDetails')

        # If they choose to login, send them to the login page
        if request.form["action"] == "log in":
            return redirect('/login')

        # If they choose to sign up, send them to the sign up page
        if request.form["action"] == "sign up":
            return redirect('/createUser')

        # If they want to search for a film to rate (or just for the details)
        if request.form['action'] == 'Film Search':
            # Make sure they've actually populated the search term, otherwise
            # just send them back to the home page
            if request.form['search'] != "":
                session['search_term'] = request.form['search']
                session.modified = True
                return redirect('/searchResults')

        # If they click next film, give them the next film.
        if request.form["action"] == "next film":
            session['recCount'] += 1
            # This should be pretty rare, but if a non-user exhausts the list of
            # LMR rated films (1000+...) then go from the start again.
            if session['recCount'] > len(movieRatingsList):
                session['recCount'] = 0
            session.modified = True
            return redirect('/')

    f = open('../HTML/lw01_user_homepage_not_logged_in.txt', 'r')
    page = f.read()
    f.close()
    return page.format(
        film_name=film_name,
        film_year=str(film_year),
        film_desc=film_desc,
        film_genres=film_genres,
        film_photo=film_photo,
        stream_string=stream_string,
        hof1_name=hall_of_fame.index[0],
        hof1_count=hall_of_fame[hall_of_fame.index[0]],
        hof1_avg=movieRatingsList[hall_of_fame.index[0]].mean(),
        hof2_name=hall_of_fame.index[1],
        hof2_count=hall_of_fame[hall_of_fame.index[1]],
        hof2_avg=movieRatingsList[hall_of_fame.index[1]].mean(),
        hof3_name=hall_of_fame.index[2],
        hof3_count=hall_of_fame[hall_of_fame.index[2]],
        hof3_avg=movieRatingsList[hall_of_fame.index[2]].mean(),
        hof4_name=hall_of_fame.index[3],
        hof4_count=hall_of_fame[hall_of_fame.index[3]],
        hof4_avg=movieRatingsList[hall_of_fame.index[3]].mean(),
        hof5_name=hall_of_fame.index[4],
        hof5_count=hall_of_fame[hall_of_fame.index[4]],
        hof5_avg=movieRatingsList[hall_of_fame.index[4]].mean(),
        today=today,
        film_tconst=film_tconst
    )



@app.route('/login', methods=["GET", "POST"])
def login():

    '''How the user logs in'''

    # Clear the session, could be a problem with this if multiple people go on
    # the website at the same time...
    session.clear()

    # Reads in the account details file. Again saved as csv in PythonAnywhere.
    # accountDetails = pd.read_csv("accountDetails.csv")
    user_file = pd.read_csv("accountDetails.csv")
    user_file['user_lower'] = user_file['User'].astype(str).str.lower()

    if request.method == "POST":
        # If they want to set up a new user takes them to the new user screen.
        action = request.form.get("action", "")

        if action == "New User":
            return redirect('/createUser')

        # If they want to search for a film to rate (or just for the details)
        if action == 'Film Search':
            if request.form['search'] != "":
                session['search_term'] = request.form['search']
                session.modified = True
                return redirect('/searchResults')

        # If they click log in...
        elif action == "Log in":

            username = request.form["user"]
            matching_users = user_file[user_file['user_lower'] == username.lower()]

            # Checks if the user is in the database.
            if not matching_users.empty:
                # If duplicates exist (e.g. case variants), prioritize exact-case
                # match when available; otherwise use first.
                exact_case = matching_users[matching_users['User'] == username]
                user_row = exact_case.iloc[0] if not exact_case.empty else matching_users.iloc[0]

                # Checks the password listed matches the password given sends
                # the user to the menu.
                if str(user_row['Password']) == request.form["password"]:
                    session['logged_in'] = True
                    session["user"] = user_row['User']
                    session["country"] = user_row['Country']
                    session['recCount'] = 0
                    session.modified = True
                    return redirect('/menu')

                # If they got the password, lets them know and they go again.
                else:
                    f = open('../HTML/lw03_login_failed.txt', 'r')
                page = f.read()
                f.close()
                return page

            # If they got the username wrong, lets them know and they go again.
            else:
                f = open('../HTML/lw03_login_failed.txt', 'r')
                page = f.read()
                f.close()
                return page

    # The log in screen
    f = open('../HTML/lw02_log_in.txt', 'r')
    page = f.read()
    f.close()
    return page

@app.route('/createUser', methods=["GET", "POST"])
def createUser():
    '''Where the user creates their account.'''

    # Reading in the movie ratings list and account details.
    movieRatingsList = pd.read_csv("movieRatingsList.csv")
    accountDetails = pd.read_csv("accountDetails.csv")

    if request.method == "POST":

        # If the user wants to create an account.
        if request.form["action"] == "Create account":

            user = request.form["user"]
            password = request.form["password"]

            # If they entered a user who is already in the database,
            # lets them know.
            if accountDetails['User'].astype(str).str.lower().eq(request.form["user"].lower()).any():
                f = open('../HTML/lw07_sign_up_email_or_user_taken.txt', 'r')
                page = f.read()
                f.close()
                return page

            # If they left either field blank, lets them know.
            if not user or not password or not re.match(r'^[A-Za-z0-9]+$', user):
                f = open('../HTML/lw08_sign_up_user_or_pass_blank.txt', 'r')
                page = f.read()
                f.close()
                return page

            # Makes the user they have just entered the session's user -
            # essentially loggin them in.
            session['logged_in'] = True
            session['user'] = request.form['user']
            session['country'] = request.form['country']
            session.modified = True

            # Assigns them a column (a blank one) in the movieRatingsList and
            # saves it
            NaN = np.nan
            movieRatingsList[session['user']] = NaN
            movieRatingsList.to_csv("movieRatingsList.csv", index=False)

            # Setting the predicted scores as the average user rating on the site
            movieRatingsList = movieRatingsList.set_index('tconst')
            pred_scores = pd.read_csv('pred_scores.csv', index_col = 'tconst')
            pred_scores[session['user']] = movieRatingsList[movieRatingsList['NoUserInput'] == False].iloc[:,11:].mean(axis=1)
            pred_scores.to_csv('pred_scores.csv', index=True)

            # Setting the watch a film no one's seen as IMDB ratings
            fp_pred_scores = pd.read_csv('fp_pred_scores.csv', index_col = 'tconst')
            fp_pred_scores[session['user']] = movieRatingsList['averageRating']
            fp_pred_scores.to_csv('fp_pred_scores.csv', index=True)

            # Adds them to the account details csv.
            newUser = {
                'User': request.form['user'],
                'Password': request.form['password'],
                'Country': request.form['country'],
                'Email': request.form['email']}
            accountDetails = pd.concat(
                [accountDetails, pd.DataFrame([newUser])],
                ignore_index=True)
            accountDetails.to_csv("accountDetails.csv", index=False)

            # Takes them to the main menu.
            return redirect('/menu')

        # If they want to search for a film to rate (or just for the details)
        if request.form['action'] == 'Film Search':
            if request.form['search'] != "":
                session['search_term'] = request.form['search']
                session.modified = True
                return redirect('/searchResults')

        # Allows them to go back to the login menu.
        elif request.form["action"] == "log in":
            return redirect('/login')

        # Clicking on the sign up button just brings them back to the same page.
        elif request.form["action"] == "sign up":
            return redirect('/createUser')

    # New user menu.
    f = open('../HTML/lw06_sign_up.txt', 'r')
    page = f.read()
    f.close()
    return page

@app.route('/menu', methods=["GET", "POST"])
def menu():
    '''The home page once the user has logged in. Expands on the non-logged in
    home page by being more personalised'''

    if session.get('logged_in') == False:
        return redirect('/')

    # Pulling the movieRatingsList from the saved csv file.
    movieRatingsList = pd.read_csv(
        "movieRatingsList.csv",
        index_col='tconst')

    # Reading in the projected recommendations list.
    projRecList = pd.read_csv(
        "pred_scores.csv",
        index_col='tconst')

    # The user is the sessions user.
    user = session["user"]

    # Finding the specific users projected recommendations.
    userProjRecList = projRecList[projRecList[user].notnull()].sort_values(
        user,
        ascending=False)
    userProjRecList = userProjRecList.merge(
        movieRatingsList,
        left_index=True,
        right_index=True,
        how='inner')[[
            'primaryTitle',
            'genre1',
            'genre2',
            'genre3',
            'startYear',
            'runtimeMinutes']]

    null_rating_warning = ''

    # Menu option redirections.
    if request.method == "POST":

        # If the user clicks on any of the film images, they will be sent to
        # the films detail page.
        tconst = request.form.get("tconst")
        if tconst:
            session['tconst'] = tconst
            session.modified = True
            return redirect('/filmDetails')

        # If the user wants to log out, clear the session and send them back to
        # the non-logged in page.
        if request.form["action"] == "log out":
            session.clear()
            return redirect('/')

        # If they want to search for a film to rate (or just for the details)
        if request.form['action'] == 'Film Search':
            # Make sure they have populated the search bar, otherwise ust send
            # them back to the home page
            if request.form['search'] != "":
                session['search_term'] = request.form['search']
                session.modified = True
                return redirect('/searchResults')

        # if they want to enter a rating
        if request.form["action"] == "enter rating":
            # Make sure the rating is above 0 and equal to or below 10
            rating_value = float(request.form["amountRange"])
            if (rating_value > 0) and (rating_value <= 10):
                # Change the value in the movie ratings list to the rating given
                # by the user
                movieRatingsList.at[
                    userProjRecList.index[session['recCount']],
                    user] = rating_value

                # Make sure that the NoUserInput field is set to False (because
                # a user has definitely now rated the film). On the home screen
                # this should always be the case, but just in case there's any
                # funny business.
                movieRatingsList.at[
                    userProjRecList.index[session['recCount']],
                    'NoUserInput'] = False

                # Save the movie ratings list
                movieRatingsList.to_csv('movieRatingsList.csv', index=True)

                # In the projected ratings list, change the vallue to NaN so the
                # user will no longer be recommended the film, save the file and
                # regenerate the users recommendations.
                projRecList.at[userProjRecList.index[session['recCount']], user] = np.nan
                projRecList.to_csv('pred_scores.csv', index=True)
                userProjRecList = projRecList[projRecList[user].notnull()].sort_values(
                    user,
                    ascending=False)
                userProjRecList = userProjRecList.merge(
                    movieRatingsList,
                    left_index=True,
                    right_index=True,
                    how='inner')[[
                        'primaryTitle',
                        'genre1',
                        'genre2',
                        'genre3',
                        'startYear',
                        'runtimeMinutes']]
            # If the rating is out of the range, let the user know.
            else:
                null_rating_warning = '<p class="error-text">Enter a rating greater than 0 and less than or equal to 10</p> <!-- Error message -->'

        # If they want the next film, give it to them.
        if request.form["action"] == "next film":
            session['recCount'] += 1
            # If they've run out of films, start them from the start again.
            if session['recCount'] > len(userProjRecList):
                session['recCount'] = 0
            session.modified = True

        # If the user wants to go back to the menu, reset the recCount and send
        # them back.
        elif request.form['action'] == 'home':
            session['recCount'] = 0
            session.modified = True
            return redirect('/menu')

        # If they want to rate a film.
        elif request.form["action"] == "Rate a Film":
            return redirect('/newRateMostVoted')

        # If they want to rate a film no one on the DB has rated.
        elif request.form["action"] == "Rate a New Film":
            return redirect('/newRateMostVotedNew')

        # If they want to apply a filter.
        elif request.form['action'] == 'Feeling Picky':
            session["recCount"] = 0
            session.modified = True
            return redirect('/newFilmFilter')

        # If they want to watch with a friend.
        elif request.form['action'] == "Feeling Friendly":
            return redirect('/newPartnerFind')

        # If they want to watch with a friend.
        elif request.form['action'] == "Feeling Pretentious":
            return redirect('/feelingPretentious')

        # If they want to merge Letterboxd ratings
        elif request.form['action'] == "Import Letterboxd":
            return redirect('/lboxUpload')

    # If they haven't clicked next next film then this is their first rec.
    if 'recCount' not in session:
        session['recCount'] = 0

    # Getting info about the current film to display
    film_tconst = str(userProjRecList.index[session['recCount']])
    film_name = str(userProjRecList.loc[film_tconst,'primaryTitle'])
    film_year = userProjRecList.loc[film_tconst,'startYear']
    film_genres = ' • '.join(userProjRecList.loc[
        film_tconst,
        ['genre1','genre2','genre3']].dropna().astype(str))

    try: film_desc = descFilm(str(film_name), int(film_year))
    except: film_desc = ''
    try: film_photo = filmPhoto(film_name, film_year)
    except: film_photo = ''

    # Geting streams
    try: film_streams = filmStreams(film_name, film_year, session['country'])
    except: film_streams = []

    stream_string = ''
    for stream in film_streams:
        imageLink = streamImagesLinks(stream, 'AU')
        if imageLink:
            stream_string += ''' <a href="{link}"><img src="{photo}" style="max-width: 40px; height: auto; margin-bottom: 4px;" alt="{stream}"></a>'''.format(photo=imageLink[0], stream=stream, link=imageLink[1])

    # Getting today's day of the week - literally for the one sentence at the
    # bottom of the home page
    today = datetime.date.today().strftime("%A")


    # Building the hall of fame
    user_summary = movieRatingsList.iloc[:,10:].agg(['count', 'mean']).transpose()

    # Renaming the user as 'You'
    user_summary.rename(index={user: 'You'}, inplace=True)
    user_summary.sort_values(by='count', ascending=False, inplace=True)

    # Getting the top 5 'hall of famers'
    top5 = user_summary.sort_values(by='count', ascending=False)[:5].index

    # If the user is in there, we want them listed in the top 5.
    if 'You' in top5:

        rankings_html = f'''<div class="w3-half">
      					<div class="w3-card w3-container" style="margin: 10px; min-height: 56vh;">
        						<h2><b>Hall of Fame</b></h2><br>
        						<table style="max-width: 100%; table-layout: auto;">
          							<tr>
            							<th style="text-align: center; font-size: 20px; width: 20%;"></th>
            							<th style="text-align: center; font-size: 20px; width: 27%;">User</th>
            							<th style="text-align: center; font-size: 20px; width: 24%;">Films Rated</th>
    								<th style="text-align: center; font-size: 20px; width: 29%;">Average Rating</th>
          							</tr>
          							<tr>
            							<td style="text-align: center; font-size: 36px; width: 20%;"><img src="https://ew.com/thmb/12ygs-8rgSUhb4qi11u3_ze1V40=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/sly-stallone-philly-tout-120423-df615c4fd7c94c99a3d1cbd70a662a88.jpg" alt="1." style="width:80%"></td>
            							<td style="text-align: center; font-size: 36px; width: 27%;"><strong>{user_summary.index[0]}</strong></td>
            							<td style="text-align: center; font-size: 36px; width: 24%;">{user_summary.iloc[0,0]:.0f}</td>
    								<td style="text-align: center; font-size: 36px; width: 29%;">{user_summary.iloc[0,1]:.1f}</td>
          							</tr>
          							<tr>
            							<td style="text-align: center; font-size: 30px; width: 20%;">2.</td>
            							<td style="text-align: center; font-size: 30px; width: 27%;"><strong>{user_summary.index[1]}</strong></td>
            							<td style="text-align: center; font-size: 30px; width: 24%;">{user_summary.iloc[1,0]:.0f}</td>
    								<td style="text-align: center; font-size: 30px; width: 29%;">{user_summary.iloc[1,1]:.1f}</td>
          							</tr>
          							<tr>
            							<td style="text-align: center; font-size: 24px; width: 20%;">3.</td>
            							<td style="text-align: center; font-size: 24px; width: 27%;"><strong>{user_summary.index[2]}</strong></td>
            							<td style="text-align: center; font-size: 24px; width: 24%;">{user_summary.iloc[2,0]:.0f}</td>
    								<td style="text-align: center; font-size: 24px; width: 29%;">{user_summary.iloc[2,1]:.1f}</td>
          							</tr>
          							<tr>
            							<td style="text-align: center; font-size: 18px; width: 20%;">4.</td>
            							<td style="text-align: center; font-size: 18px; width: 27%;"><strong>{user_summary.index[3]}</strong></td>
            							<td style="text-align: center; font-size: 18px; width: 24%;">{user_summary.iloc[3,0]:.0f}</td>
    								<td style="text-align: center; font-size: 18px; width: 29%;">{user_summary.iloc[3,1]:.1f}</td>
          							</tr>
          							<tr>
            							<td style="text-align: center; font-size: 14px; width: 20%;">5.</td>
            							<td style="text-align: center; font-size: 14px; width: 27%;"><strong>{user_summary.index[4]}</strong></td>
            							<td style="text-align: center; font-size: 14px; width: 24%;">{user_summary.iloc[4,0]:.0f}</td>
    								<td style="text-align: center; font-size: 14px; width: 29%;">{user_summary.iloc[4,1]:.1f}</td>
          							</tr>
        						</table>
      					</div>
    				</div>'''

    # If they are not, then they should still be shown, but will be lower.
    else:
        rankings_html = f'''<div class="w3-half">
      					<div class="w3-card w3-container" style="margin: 10px; min-height: 56vh;">
        						<h2><b>Hall of Fame</b></h2><br>
        						<table style="max-width: 100%; table-layout: auto;">
          							<tr>
            							<th style="text-align: center; font-size: 20px; width: 20%;"></th>
            							<th style="text-align: center; font-size: 20px; width: 27%;">User</th>
            							<th style="text-align: center; font-size: 20px; width: 24%;">Films Rated</th>
    								<th style="text-align: center; font-size: 20px; width: 29%;">Average Rating</th>
          							</tr>
          							<tr>
            							<td style="text-align: center; font-size: 36px; width: 20%;"><img src="https://ew.com/thmb/12ygs-8rgSUhb4qi11u3_ze1V40=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/sly-stallone-philly-tout-120423-df615c4fd7c94c99a3d1cbd70a662a88.jpg" alt="1." style="width:80%"></td>
            							<td style="text-align: center; font-size: 36px; width: 27%;"><strong>{user_summary.index[0]}</strong></td>
            							<td style="text-align: center; font-size: 36px; width: 24%;">{user_summary.iloc[0,0]:.0f}</td>
    								<td style="text-align: center; font-size: 36px; width: 29%;">{user_summary.iloc[0,1]:.1f}</td>
          							</tr>
          							<tr>
            							<td style="text-align: center; font-size: 30px; width: 20%;">2.</td>
            							<td style="text-align: center; font-size: 30px; width: 27%;"><strong>{user_summary.index[1]}</strong></td>
            							<td style="text-align: center; font-size: 30px; width: 24%;">{user_summary.iloc[1,0]:.0f}</td>
    								<td style="text-align: center; font-size: 30px; width: 29%;">{user_summary.iloc[1,1]:.1f}</td>
          							</tr>
          							<tr>
            							<td style="text-align: center; font-size: 24px; width: 20%;">3.</td>
            							<td style="text-align: center; font-size: 24px; width: 27%;"><strong>{user_summary.index[2]}</strong></td>
            							<td style="text-align: center; font-size: 24px; width: 24%;">{user_summary.iloc[2,0]:.0f}</td>
    								<td style="text-align: center; font-size: 24px; width: 29%;">{user_summary.iloc[2,1]:.1f}</td>
          							</tr>
          							<tr>
            							<td style="text-align: center; font-size: 18px; width: 20%;">4.</td>
            							<td style="text-align: center; font-size: 18px; width: 27%;"><strong>{user_summary.index[3]}</strong></td>
            							<td style="text-align: center; font-size: 18px; width: 24%;">{user_summary.iloc[3,0]:.0f}</td>
    								<td style="text-align: center; font-size: 18px; width: 29%;">{user_summary.iloc[3,1]:.1f}</td>
          							</tr>
                                    <tr>
            							<td style="text-align: center; font-size: 20px; width: 20%;">&#x22EE</td>
            							<td style="text-align: center; font-size: 20px; width: 27%;">&#x22EE</td>
            							<td style="text-align: center; font-size: 20px; width: 24%;">&#x22EE</td>
    								    <td style="text-align: center; font-size: 20px; width: 29%;">&#x22EE</td>
          							</tr>
          							<tr>
            							<td style="text-align: center; font-size: 14px; width: 20%;">{user_summary.rank(method='min', ascending=False).loc['You', 'count']:.0f}.</td>
            							<td style="text-align: center; font-size: 14px; width: 27%;"><strong>You</strong></td>
            							<td style="text-align: center; font-size: 14px; width: 24%;">{user_summary.loc['You','count']:.0f}</td>
    								<td style="text-align: center; font-size: 14px; width: 29%;">{user_summary.loc['You','mean']:.1f}</td>
          							</tr>
        						</table>
      					</div>
    				</div>'''

    # Genre symbols, currently (and probably future) hard coded
    genre_symbols = {
        'Reality-TV': "fa fa-star",
        'Mystery': "fas fa-search",
        'Comedy': "fa-solid fa-face-grin-tears",
        'Western': "fas fa-hat-cowboy",
        'Family': "fa-solid fa-children",
        'Crime': "fa-solid fa-handcuffs",
        'Music': "fa-solid fa-music",
        'Fantasy': "fa-solid fa-dragon",
        'Film-Noir': 'fa-solid fa-user-secret',
        'War': 'fa-solid fa-person-rifle',
        'Sci-Fi': 'fa-solid fa-hand-spock',
        'Horror': 'fa-solid fa-ghost',
        'Thriller': 'fa-solid fa-face-surprise',
        'Adult': 'fa-solid fa-face-kiss-wink-heart',
        'Musical': "fa-solid fa-music",
        'Biography': "fas fa-portrait",
        'News': "fa-solid fa-newspaper",
        'Drama': "fa-solid fa-masks-theater",
        'Action': "fa-solid fa-gun",
        'Sport': "fa fa-soccer-ball-o",
        'Animation': "fa-solid fa-hippo",
        'Romance': "fa-solid fa-heart",
        'Adventure': "fa-solid fa-person-hiking",
        'Documentary': "fa-solid fa-video",
        'History': "fas fa-landmark"
    }

    personalised_str = ''

    # Only personalising if the user has actually rated a film
    if movieRatingsList[user].count() > 0:


        # Finding the users favourite film.
        fav_details = movieRatingsList.sort_values(
            by=[user, 'numVotes'],
            ascending=[False,True]).reset_index().loc[0,:]
        fav_tconst = fav_details.loc['tconst']
        fav_name = fav_details.loc['primaryTitle']
        fav_year = fav_details.loc['startYear']
        fav_photo = filmPhoto(fav_name, fav_year)
        fav_rate = fav_details.loc[user]
        fav_avg = pd.to_numeric(fav_details[11:], errors='coerce').mean()

        # Finding the user's least favourite film
        hate_details = movieRatingsList.sort_values(
            by=[user, 'numVotes'],
            ascending=[True,False]).reset_index().loc[0,:]
        hate_tconst = hate_details.loc['tconst']
        hate_name = hate_details.loc['primaryTitle']
        hate_year = hate_details.loc['startYear']
        hate_photo = filmPhoto(hate_name, hate_year)
        hate_rate = hate_details.loc[user]
        hate_avg = pd.to_numeric(hate_details[11:], errors='coerce').mean()

        # Finding the user that is most similar
        ideal_partner = most_sim_user(movieRatingsList, user)

        # Finding the user's top 5 favourite genres
        top_5 = top_5_genres(movieRatingsList, user)

        # Generating a spider chart of the user's genre profile
        genre_spider_chart(build_genre_ratings_df(
            movieRatingsList,
            [user,ideal_partner],
            True),
            '../Python/static')

        # Creating the HTML for the bottom 'personalised' half of the page.
        personalised_str = f'''<div class="w3-row-padding w3-center w3-margin-top">
      					<div class="w3-third">
        						<div class="w3-card w3-container" style="margin: 10px; min-height: 36vh;">
          							<h3><b>Your Favourite Film</b></h3><br>
          							<div class="w3-third">
          							    <form method="post" action="/menu">
                                            <input type="hidden" name="tconst" value="{fav_tconst}">
                                            <button type="submit" style="border: none; background: none; padding: 0; cursor: pointer;">
            							        <img src="{fav_photo}" alt="Favourite Film" style="width: 75%; max-width: 75%; height: auto;">
            							    </button>
                                        </form>
          							</div>
          							<div class="w3-twothird">
            							<h4><b>{fav_name} ({fav_year})</b></h4>
            							<p style="text-align:centre">Your rating: <strong>{fav_rate:.1f}</strong></p>
            							<p style="text-align:centre">Average LMR rating: <strong>{fav_avg:.1f}</strong></p>
          							</div>
        						</div>
        						<div class="w3-card w3-container" style="margin: 10px; min-height: 31vh;">
          							<h3><b>Your ideal viewing partner is...</b></h3><br>
          							<h1><b>{ideal_partner}!</b></h1>
        						</div>
      					</div>
      					<div class="w3-third">
        						<div class="w3-card w3-container" style="margin: 10px; min-height: 68vh; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
          							<h3><b>Your Genre Profile</b></h3><br>
          							<img src="/static/spider_chart.png" alt="User Genre Profile" style="width: 85%; max-width: 85%; height: auto;">
        						</div>
      					</div>
      					<div class="w3-third">
        						<div class="w3-card w3-container" style="margin: 10px; min-height: 36vh;">
          							<h3><b>Your Most Hated Film</b></h3><br>
          							<div class="w3-third">
          							    <form method="post" action="/menu">
                                            <input type="hidden" name="tconst" value="{hate_tconst}">
                                            <button type="submit" style="border: none; background: none; padding: 0; cursor: pointer;">
            							        <img src="{hate_photo}" alt="Car" style="width: 75%; max-width: 75%; height: auto;">
            							    </button>
                                        </form>
          							</div>
          							<div class="w3-twothird">
            							<h4><b>{hate_name} ({hate_year})</b></h4>
            							<p style="text-align:centre">Your rating: <strong>{hate_rate:.1f}</strong></p>
            							<p style="text-align:centre">Average LMR rating: <strong>{hate_avg:.1f}</strong></p>
          							</div>
        						</div>
        						<div class="w3-card w3-container" style="margin: 10px; min-height: 31vh;">
                                    <h3><b>Your Top Genres</b></h3><br>
                                    <div class="w3-row" style="display: flex; justify-content: space-between; align-items: center; text-align: center; flex-wrap: wrap;">
                                        <div class="w3-col" style="display: flex; flex-direction: column; align-items: center; width: 20%;">
                                            <i class="{genre_symbols[top_5.index[0]]}" style="font-size: 40px;"></i>
                                            <p style="font-size: 12px;"><b>{top_5.index[0]}</b></p>
                                            <p><b>{top_5[0]:.1f}</b></p>
                                        </div>
                                        <div class="w3-col" style="display: flex; flex-direction: column; align-items: center; width: 20%;">
                                            <i class="{genre_symbols[top_5.index[1]]}" style="font-size: 40px;"></i>
                                            <p style="font-size: 12px;"><b>{top_5.index[1]}</b></p>
                                            <p><b>{top_5[1]:.1f}</b></p>
                                        </div>
                                        <div class="w3-col" style="display: flex; flex-direction: column; align-items: center; width: 20%;">
                                            <i class="{genre_symbols[top_5.index[2]]}" style="font-size: 40px;"></i>
                                            <p style="font-size: 12px;"><b>{top_5.index[2]}</b></p>
                                            <p><b>{top_5[2]:.1f}</b></p>
                                        </div>
                                        <div class="w3-col" style="display: flex; flex-direction: column; align-items: center; width: 20%;">
                                            <i class="{genre_symbols[top_5.index[3]]}" style="font-size: 40px;"></i>
                                            <p style="font-size: 12px;"><b>{top_5.index[3]}</b></p>
                                            <p><b>{top_5[3]:.1f}</b></p>
                                        </div>
                                        <div class="w3-col" style="display: flex; flex-direction: column; align-items: center; width: 20%;">
                                            <i class="{genre_symbols[top_5.index[4]]}" style="font-size: 40px;"></i>
                                            <p style="font-size: 12px;"><b>{top_5.index[4]}</b></p>
                                            <p><b>{top_5[4]:.1f}</b></p>
                                        </div>
                                    </div>
                                </div>
'''


    # The logged in home page.
    f = open('../HTML/lw04_user_homepage_logged_in.txt', 'r')
    page = f.read()
    f.close()
    return page.format(
        rankings_html=rankings_html,
        film_photo=film_photo,
        stream_string=stream_string,
        film_name=film_name,
        film_year=film_year,
        film_desc=film_desc,
        film_genres=film_genres,
        null_rating_warning=null_rating_warning,
        personalised_str=personalised_str,
        today=today,
        user=user,
        film_tconst=film_tconst)

@app.route('/searchResults', methods=["GET", "POST"])
def searchResults():
    '''The results of the users search, any film that is a close enough match is
    listed in the results.'''

    # Reading in the movie ratings list
    movieRatingsList = pd.read_csv("movieRatingsList.csv")

    if request.method == "POST":

        # If the user clicks on a film, take them to the details of that film
        tconst = request.form.get("tconst")
        if tconst:
            session['tconst'] = tconst
            session.modified = True
            return redirect('/filmDetails')

    # If they are logged in give them the logged in side menu
    if session['logged_in'] == True:
        sidebar_menu = f'''<!-- Sidebar/menu -->
		<nav class="w3-sidebar w3-white w3-animate-left" style="z-index:3;width:300px; display: none;" id="mySidebar">
  			<div class="w3-container w3-row" style="position: relative; padding-top: 10px;">
    				<div class="w3-col s8 w3-bar" style="padding-top: 10px;">
      					<span>Welcome back, <strong>{session['user']}</strong>!</span>
					<p></p>
					<form method="post" action="/menu">
					<button class="w3-button w3-khaki" id="submit" name="action" value="log out">Log Out</button>
					</form>
    				</div>
    				<div class="w3-col s4" style="text-align: right; padding-top: 10px;">
      					<button style="width: 30px; height: 30px; padding: 0; margin: 0;" name="action" class="w3-button w3-white w3-hover-grey" onclick="w3_close()">
        					<i class="fa fa-remove" style="font-size: 24px;"></i>
      					</button>
    				</div>
  			</div>
  			<hr>
  				<div class="w3-bar-block">
					<div class="w3-bar-item" style="display: flex; align-items: center;">
                        <form method="post" action="/menu" style="display: flex; width: 100%;">
                        <input onkeyup="if(event.keyCode===13){{submit.click()}}"
                            style="flex-grow: 1; border: 1px solid #f0e68c; padding: 8px;"
                            type="text"
                            class="w3-bar-item"
                            placeholder="Film Search..."
                            name="search">
                            <button id="submit"
                                type="submit"
                                name="action"
                                class="w3-bar-item w3-khaki w3-hover-white"
                                value="Film Search"
                                style="margin-left: 5px; width: 40px; height: 40px; display: flex; justify-content: center; align-items: center;">
                            <i class="fa fa-search"></i>
                            </button>
                        </form>
                    </div>

					    <form method="post" action="/menu">
    					<button class="w3-bar-item w3-button w3-padding w3-khaki" id="submit" name="action" value="home"><i class="fa fa-home fa-fw"></i>  Home</button>
    					<button class="w3-bar-item w3-button w3-padding w3-khaki" id="submit" name="action" value="Rate a Film"><i class="fa fa-film fa-fw"></i>  Rate Films</button>
    					<button class="w3-bar-item w3-button w3-padding w3-khaki" id="submit" name="action" value="Rate a New Film"><i class="fa fa-eye fa-fw"></i>  Rate New Films</button>
    					<button class="w3-bar-item w3-button w3-padding w3-khaki" id="submit" name="action" value="Feeling Friendly"><i class="fa fa-users fa-fw"></i>  Feeling Friendly</button>
    					<button class="w3-bar-item w3-button w3-padding w3-khaki" id="submit" name="action" value="Feeling Picky"><i class="fa fa-bullseye fa-fw"></i>  Feeling Picky</button>
					    <button class="w3-bar-item w3-button w3-padding w3-khaki" id="submit" name="action" value="Feeling Pretentious"><i class="fa-solid fa-camera-retro fa-fw"></i>  Feeling Pretentious</button>
					    <button class="w3-bar-item w3-button w3-padding w3-khaki" id="submit" name="action" value="Import Letterboxd"><i class="fa-brands fa-square-letterboxd"></i>  Import Letterboxd Ratings</button>
					</form>
  				</div>
			</div>
		</nav>
'''
        country = session['country']

    # Otherwise given them the not logged in verion of the side menu
    else:
        sidebar_menu = '''<!-- Sidebar/menu -->
		<nav class="w3-sidebar w3-white w3-animate-left" style="z-index:3;width:300px; display: none;" id="mySidebar">
  			<div class="w3-container w3-row" style="position: relative; padding-top: 10px;">
    				<div class="w3-col s8 w3-bar" style="padding-top: 10px;">
    				    <form method="post" action=".">
					        <button class="w3-button w3-khaki" name="action" value="log in">Log In</button><button class="w3-button w3-khaki" name="action" value="sign up">Sign Up</button>
    				    </form>
    				</div>
    				<div class="w3-col s4" style="text-align: right; padding-top: 10px;">
      					<button style="width: 30px; height: 30px; padding: 0; margin: 0;" name="action" class="w3-button w3-white w3-hover-grey" onclick="w3_close()">
        					<i class="fa fa-remove" style="font-size: 24px;"></i>
      					</button>
    				</div>
  			</div>
  			<hr>
  				<div class="w3-bar-block">
					<div class="w3-bar-item" style="display: flex; align-items: center;">
                        <form method="post" action="/menu" style="display: flex; width: 100%;">
                            <input onkeyup="if(event.keyCode===13){{submit.click()}}"
                                style="flex-grow: 1; border: 1px solid #f0e68c; padding: 8px;"
                                type="text"
                               class="w3-bar-item"
                               placeholder="Film Search..."
                               name="search">
                            <button id="submit"
                                type="submit"
                                name="action"
                                class="w3-bar-item w3-khaki w3-hover-white"
                                value="Film Search"
                                style="margin-left: 5px; width: 40px; height: 40px; display: flex; justify-content: center; align-items: center;">
                                <i class="fa fa-search"></i>
                            </button>
                        </form>
                    </div>
					    <form method="post" action=".">
    					<button class="w3-bar-item w3-button w3-padding w3-khaki" id="submit" name="action" value="home"><i class="fa fa-home fa-fw"></i>  Home</button>
    					</form>
  				</div>
			</div>
		</nav>
'''
        country = 'AU'

    # Find the top 100 films based on the search term
    search_results = pd.DataFrame(process.extract(
        session['search_term'],
        movieRatingsList['primaryTitle'].unique(),
        limit=100), columns=['primaryTitle', 'search_score'])
    # Only return those with a score of 75 or higher
    search_results = search_results[search_results['search_score'] >= 75]
    search_results = movieRatingsList[[
        'tconst',
        'primaryTitle',
        'startYear',
        'genre1',
        'genre2',
        'genre3',
        'numVotes']].merge(search_results, how='inner', on='primaryTitle')
    # Sort by the closest match
    search_results.sort_values(
        ['search_score', 'numVotes'],
        ascending=False,
        inplace=True)
    search_out = ''

    # Getting the details of the films
    for r in search_results.itertuples(index=True, name='Row'):

        film_tconst = r.tconst
        film_name = r.primaryTitle
        film_year = r.startYear
        # The film doesn't always have a description or a photo, in those
        # instances, just make them blank.
        try: film_desc = descFilm(str(film_name), int(film_year))
        except: film_desc = ''
        try: film_photo = filmPhoto(film_name, film_year)
        except: film_photo = ''
        film_genres = pd.Series([r.genre1, r.genre2, r.genre3]).str.cat(sep=' • ')

        # Geting streams
        try: film_streams = filmStreams(film_name, film_year, country)
        except: film_streams = []

        stream_string = ''
        for stream in film_streams:
            imageLink = streamImagesLinks(stream, country)
            if imageLink:
                stream_string += ''' <a href="{link}"><img src="{photo}" style="max-width: 40px; height: auto; margin-bottom: 4px;" alt="{stream}"></a>'''.format(
                    photo=imageLink[0],
                    stream=stream,
                    link=imageLink[1])


        search_out += f'''
        <div class="w3-row-padding w3-center w3-margin-top">
            <div class="w3-card w3-container" style="margin: 10px; min-height: 20vh;">
                <div class="film-info">
                    <div class="film-image">
                        <form method="post" action="/searchResults">
                            <input type="hidden" name="tconst" value="{film_tconst}">
                            <button type="submit" style="border: none; background: none; padding: 0; cursor: pointer;">
                                <img src="{film_photo}" alt="{film_name}" style="width: 70%; max-width: 70%; height: auto;">
                            </button>
                        </form>
                        <p class="w3-text-black" style="text-align: center;">
                            {stream_string}
                        </p>
                    </div>
                    <div class="film-details">
                        <h4 class="film-title"><b>{film_name} ({film_year})</b></h4>
                        <p class="film-description" style="text-align: left;">{film_desc}</p>
                        <p class="film-genres">{film_genres}</p>
                    </div>
                </div>
            </div>
        </div>

    '''

    # Showing the search results
    f = open('../HTML/lw10_search_results.txt', 'r')
    page = f.read()
    f.close()
    return page.format(sidebar_menu=sidebar_menu, search_out=search_out)


@app.route('/filmDetails', methods=["GET", "POST"])
def filmDetails():
    '''Details of the film the user has clicked on.'''

    # Reading in the movieratings list and getting the details of the film.
    movieRatingsList = pd.read_csv("movieRatingsList.csv",index_col='tconst')
    film_detail = movieRatingsList.loc[session['tconst']]
    film_name = film_detail['primaryTitle']
    film_year = film_detail['startYear']
    try: film_desc = descFilm(str(film_name), int(film_year))
    except: film_desc = ''
    try: film_photo = filmPhoto(film_name, film_year)
    except: film_photo = ''
    film_genres = ' • '.join(film_detail[6:9].dropna().astype(str))



    # Getting the LMR rating average, if no one has rated the film, don't put
    # the rating.
    film_average = pd.to_numeric(film_detail[10:], errors='coerce').mean()
    film_average_str = ''
    if not np.isnan(film_average):
        film_average_str = f'''&emsp;LMR Average: <b>{film_average:.1f}</b>'''

    # Get the IMDB rating
    film_imdb = film_detail['averageRating']

    if request.method == "POST":

        # If the user wants to rate the film
        if request.form["action"] == "enter rating":

            # Make sure the rating is above 0 and less than or equal to 10
            rating_value = float(request.form["amountRange"])
            if (rating_value > 0) and (rating_value <= 10):

                # Change the users rating in the movie ratings list and set
                # nouserinput to False (a user has now rated it).
                movieRatingsList.at[
                    session['tconst'],
                    session['user']] = rating_value
                movieRatingsList.at[session['tconst'], 'NoUserInput'] = False
                movieRatingsList.to_csv('movieRatingsList.csv', index=True)

                # Read in the projected recommendation list, set the user's
                # pred score to NaN - so they don't get recommended the film.
                projRecList = pd.read_csv("pred_scores.csv")
                projRecList.loc[projRecList['tconst'] == session['tconst'], session['user']] = np.nan
                projRecList.to_csv('pred_scores.csv', index=False)

                # Update the details of the film.
                film_detail = movieRatingsList.loc[session['tconst']]
                film_name = film_detail['primaryTitle']
                film_year = film_detail['startYear']
                try: film_desc = descFilm(str(film_name), int(film_year))
                except: film_desc = ''
                try: film_photo = filmPhoto(film_name, film_year)
                except: film_photo = ''
                film_genres = ' • '.join(film_detail[6:9].dropna().astype(str))

                film_average = pd.to_numeric(film_detail[10:], errors='coerce').mean()
                film_average_str = ''
                if not np.isnan(film_average):
                    film_average_str = f'''&emsp;LMR Average: <b>{film_average:.1f}</b>'''

                film_imdb = film_detail['averageRating']

            # If the rating is out of the range, let the user know.
            else:
                null_rating_warning = '<p class="error-text">Enter a rating greater than 0 and less than or equal to 10</p> <!-- Error message -->'


    # Default values for the HTML strings
    user_rating_str = ''
    null_rating_warning = ''
    rating_entry = ''

    # If the user is logged in, give them the logged in side menu
    if session['logged_in'] == True:
        sidebar_menu = f'''<!-- Sidebar/menu -->
		<nav class="w3-sidebar w3-white w3-animate-left" style="z-index:3;width:300px; display: none;" id="mySidebar">
  			<div class="w3-container w3-row" style="position: relative; padding-top: 10px;">
    				<div class="w3-col s8 w3-bar" style="padding-top: 10px;">
      					<span>Welcome back, <strong>{session['user']}</strong>!</span>
					<p></p>
					<form method="post" action="/menu">
					<button class="w3-button w3-khaki" id="submit" name="action" value="log out">Log Out</button>
					</form>
    				</div>
    				<div class="w3-col s4" style="text-align: right; padding-top: 10px;">
      					<button style="width: 30px; height: 30px; padding: 0; margin: 0;" name="action" class="w3-button w3-white w3-hover-grey" onclick="w3_close()">
        					<i class="fa fa-remove" style="font-size: 24px;"></i>
      					</button>
    				</div>
  			</div>
  			<hr>
  				<div class="w3-bar-block">
					<div class="w3-bar-item" style="display: flex; align-items: center;">
                        <form method="post" action="/menu" style="display: flex; width: 100%;">
                        <input onkeyup="if(event.keyCode===13){{submit.click()}}"
                            style="flex-grow: 1; border: 1px solid #f0e68c; padding: 8px;"
                            type="text"
                            class="w3-bar-item"
                            placeholder="Film Search..."
                            name="search">
                            <button id="submit"
                                type="submit"
                                name="action"
                                class="w3-bar-item w3-khaki w3-hover-white"
                                value="Film Search"
                                style="margin-left: 5px; width: 40px; height: 40px; display: flex; justify-content: center; align-items: center;">
                            <i class="fa fa-search"></i>
                            </button>
                        </form>
                    </div>

					    <form method="post" action="/menu">
    					<button class="w3-bar-item w3-button w3-padding w3-khaki" id="submit" name="action" value="home"><i class="fa fa-home fa-fw"></i>  Home</button>
    					<button class="w3-bar-item w3-button w3-padding w3-khaki" id="submit" name="action" value="Rate a Film"><i class="fa fa-film fa-fw"></i>  Rate Films</button>
    					<button class="w3-bar-item w3-button w3-padding w3-khaki" id="submit" name="action" value="Rate a New Film"><i class="fa fa-eye fa-fw"></i>  Rate New Films</button>
    					<button class="w3-bar-item w3-button w3-padding w3-khaki" id="submit" name="action" value="Feeling Friendly"><i class="fa fa-users fa-fw"></i>  Feeling Friendly</button>
    					<button class="w3-bar-item w3-button w3-padding w3-khaki" id="submit" name="action" value="Feeling Picky"><i class="fa fa-bullseye fa-fw"></i>  Feeling Picky</button>
					    <button class="w3-bar-item w3-button w3-padding w3-khaki" id="submit" name="action" value="Feeling Pretentious"><i class="fa-solid fa-camera-retro fa-fw"></i>  Feeling Pretentious</button>
					    <button class="w3-bar-item w3-button w3-padding w3-khaki" id="submit" name="action" value="Import Letterboxd"><i class="fa-brands fa-square-letterboxd"></i>  Import Letterboxd Ratings</button>
					</form>
  				</div>
			</div>
		</nav>
'''
        # Assign the country and user rating and let them rate the film.
        country = session['country']
        user_rating = film_detail[session['user']]
        if not np.isnan(user_rating):
            user_rating_str = f'''Your Rating: <b>{user_rating:.1f}</b>&emsp;'''

        rating_entry = f'''<form method="post" action="/filmDetails">
          								<p>Seen it? What did you think?</p>
          								{null_rating_warning}
          								<input type="range" name="amountRange" min="0" max="10" step="0.1" value="0" oninput="this.form.amountInput.value=this.value" />
          								<input type="number" name="amountInput" min="0" max="10" step="0.1" value="0" oninput="this.form.amountRange.value=this.value" />
          								<p>
            									<button class="w3-button w3-khaki" style="margin-right: 10px;" name="action" value="enter rating">Enter Rating</button>
          								</p>
        							</form>'''

    # Otherwise, don't give them their rating (there is no rating) and don't
    # let them rate (they can't)
    else:
        sidebar_menu = '''<!-- Sidebar/menu -->
		<nav class="w3-sidebar w3-white w3-animate-left" style="z-index:3;width:300px; display: none;" id="mySidebar">
  			<div class="w3-container w3-row" style="position: relative; padding-top: 10px;">
    				<div class="w3-col s8 w3-bar" style="padding-top: 10px;">
    				    <form method="post" action=".">
					        <button class="w3-button w3-khaki" name="action" value="log in">Log In</button><button class="w3-button w3-khaki" name="action" value="sign up">Sign Up</button>
    				    </form>
    				</div>
    				<div class="w3-col s4" style="text-align: right; padding-top: 10px;">
      					<button style="width: 30px; height: 30px; padding: 0; margin: 0;" name="action" class="w3-button w3-white w3-hover-grey" onclick="w3_close()">
        					<i class="fa fa-remove" style="font-size: 24px;"></i>
      					</button>
    				</div>
  			</div>
  			<hr>
  				<div class="w3-bar-block">
					<div class="w3-bar-item" style="display: flex; align-items: center;">
                        <form method="post" action="/menu" style="display: flex; width: 100%;">
                            <input onkeyup="if(event.keyCode===13){{submit.click()}}"
                                style="flex-grow: 1; border: 1px solid #f0e68c; padding: 8px;"
                                type="text"
                               class="w3-bar-item"
                               placeholder="Film Search..."
                               name="search">
                            <button id="submit"
                                type="submit"
                                name="action"
                                class="w3-bar-item w3-khaki w3-hover-white"
                                value="Film Search"
                                style="margin-left: 5px; width: 40px; height: 40px; display: flex; justify-content: center; align-items: center;">
                                <i class="fa fa-search"></i>
                            </button>
                        </form>
                    </div>
					    <form method="post" action=".">
    					<button class="w3-bar-item w3-button w3-padding w3-khaki" id="submit" name="action" value="home"><i class="fa fa-home fa-fw"></i>  Home</button>
    					</form>
  				</div>
			</div>
		</nav>
'''
        country = 'AU'

         # Geting streams
    try: film_streams = filmStreams(film_name, film_year, country)
    except: film_streams = []

    stream_string = ''
    for stream in film_streams:
        imageLink = streamImagesLinks(stream, 'AU')
        if imageLink:
            stream_string += ''' <a href="{link}"><img src="{photo}" style="max-width: 40px; height: auto; margin-bottom: 4px;" alt="{stream}"></a>'''.format(photo=imageLink[0], stream=stream, link=imageLink[1])

    # The film details page.
    f = open('../HTML/lw11_film_page.txt', 'r')
    page = f.read()
    f.close()
    return page.format(
        film_name=film_name,
        sidebar_menu=sidebar_menu,
        film_year=film_year,
        film_photo=film_photo,
        stream_string=stream_string,
        film_desc=film_desc,
        film_genres=film_genres,
        user_rating_str=user_rating_str,
        film_imdb=film_imdb,
        film_average_str=film_average_str,
        rating_entry=rating_entry)


# -------------------------- FIX UP COMMENTARY BELOW HERE

@app.route('/newRateMostVoted', methods=["GET", "POST"])
def newRateMostVoted():
    '''Gives the user a random film that is in the top 100 most popular films
    (based on number of votes) on IMDB that they haven't rated before.'''





    # Grab the movie data csv.
    movieRatingsList = pd.read_csv("movieRatingsList.csv",index_col='tconst')
    user = session["user"]
    country = session['country']

    unrated_films = movieRatingsList[movieRatingsList[user].isnull()][['primaryTitle', 'startYear', 'genre1', 'genre2', 'genre3', 'numVotes']]



    top_films = list(unrated_films.sort_values(by='numVotes', ascending=False)[:3000].index)
    random.shuffle(top_films)

    film_detail = unrated_films.loc[top_films[0],:]
    film_tconst = film_detail.name
    film_name = film_detail['primaryTitle']
    film_year = film_detail['startYear']
    film_desc = descFilm(str(film_name), int(film_year))
    film_genres = ' • '.join(film_detail[2:5].dropna().astype(str))
    film_photo = filmPhoto(film_name, film_year)

    # Geting streams
    film_streams = filmStreams(film_name, film_year, country)

    stream_string = ''
    for stream in film_streams:
        imageLink = streamImagesLinks(stream, country)
        if imageLink:
            stream_string += ''' <a href="{link}"><img src="{photo}" style="max-width: 40px; height: auto; margin-bottom: 4px;" alt="{stream}"></a>'''.format(photo=imageLink[0], stream=stream, link=imageLink[1])


    null_rating_warning = ''

    if request.method == "POST":
        if request.form["action"] == "enter rating":
            rating_value = float(request.form["amountRange"])
            if rating_value > 0 and rating_value <= 10:
                movieRatingsList.at[session['tconst'], session['user']] = rating_value
                movieRatingsList.at[session['tconst'], 'NoUserInput'] = False
                movieRatingsList.to_csv('movieRatingsList.csv', index=True)

                projRecList = pd.read_csv("pred_scores.csv")
                projRecList.loc[projRecList['tconst'] == session['tconst'], session['user']] = np.nan
                projRecList.to_csv('pred_scores.csv', index=False)

            else:
                null_rating_warning = '<p class="error-text">Enter a rating greater than 0 and less than or equal to 10</p> <!-- Error message -->'

        if request.form["action"] == "next film":
            # session['recCount'] += 1
            # session.modified = True
            return redirect('/newRateMostVoted')

    session['tconst'] = film_tconst







    # What the user sees, ask them to submit a rating on a given film.
    f = open('../HTML/lw12_film_rating_page.txt', 'r')
    page = f.read()
    f.close()
    return page.format(
        film_tconst=film_tconst,
        film_photo=film_photo,
        film_name=film_name,
        stream_string=stream_string,
        film_year=film_year,
        film_desc=film_desc,
        film_genres=film_genres,
        null_rating_warning=null_rating_warning,
        user=user)

# Up to 200???
@app.route('/newRateMostVotedNew', methods=["GET", "POST"])
def newRateMostVotedNew():
    '''Give the user a random stream of films based on the top 100 films that no
    user in the database has rated.'''

    '''Gives the user a random film that is in the top 100 most popular films
    (based on number of votes) on IMDB that they haven't rated before.'''





    # Grab the movie data csv.
    movieRatingsList = pd.read_csv("movieRatingsList.csv",index_col='tconst')
    user = session["user"]
    country = session['country']

    unrated_films = movieRatingsList[(movieRatingsList[user].isnull()) & (movieRatingsList['NoUserInput']==True)][['primaryTitle', 'startYear', 'genre1', 'genre2', 'genre3', 'numVotes']]



    top_films = list(unrated_films.sort_values(by='numVotes', ascending=False)[:3000].index)
    random.shuffle(top_films)

    film_detail = unrated_films.loc[top_films[0],:]
    film_tconst = film_detail.name
    film_name = film_detail['primaryTitle']
    film_year = film_detail['startYear']
    film_desc = descFilm(str(film_name), int(film_year))
    film_genres = ' • '.join(film_detail[2:5].dropna().astype(str))
    film_photo = filmPhoto(film_name, film_year)

    try: film_desc = descFilm(str(film_name), int(film_year))
    except: film_desc = ''
    try: film_photo = filmPhoto(film_name, film_year)
    except: film_photo = ''

    # Geting streams
    try: film_streams = filmStreams(film_name, film_year, country)
    except: film_streams = []

    stream_string = ''
    for stream in film_streams:
        imageLink = streamImagesLinks(stream, country)
        if imageLink:
            stream_string += ''' <a href="{link}"><img src="{photo}" style="max-width: 40px; height: auto; margin-bottom: 4px;" alt="{stream}"></a>'''.format(photo=imageLink[0], stream=stream, link=imageLink[1])


    null_rating_warning = ''

    if request.method == "POST":
        if request.form["action"] == "enter rating":
            rating_value = float(request.form["amountRange"])
            if rating_value > 0 and rating_value <= 10:
                movieRatingsList.at[session['tconst'], session['user']] = rating_value
                movieRatingsList.at[session['tconst'], 'NoUserInput'] = False
                movieRatingsList.to_csv('movieRatingsList.csv', index=True)

                projRecList = pd.read_csv("pred_scores.csv")
                projRecList.loc[projRecList['tconst'] == session['tconst'], session['user']] = np.nan
                projRecList.to_csv('pred_scores.csv', index=False)


            else:
                null_rating_warning = '<p class="error-text">Enter a rating greater than 0 and less than or equal to 10</p> <!-- Error message -->'

        if request.form["action"] == "next film":
            # session['recCount'] += 1
            # session.modified = True
            return redirect('/newRateMostVotedNew')

    session['tconst'] = film_tconst


    # What the user sees, ask them to submit a rating on a given film.
    f = open('../HTML/lw13_new_film_rating_page.txt', 'r')
    page = f.read()
    f.close()
    return page.format(
        film_tconst=film_tconst,
        film_photo=film_photo,
        film_name=film_name,
        stream_string=stream_string,
        film_year=film_year,
        film_desc=film_desc,
        film_genres=film_genres,
        null_rating_warning=null_rating_warning,
        user=user)


@app.route('/newFilmFilter', methods=["GET", "POST"])
def newFilmFilter():
    '''Allows the user to set a few filters for the film they are looking
    for.'''

    user=session['user']


    if request.method == "POST":
        if request.form["action"] == "find_a_film":

            genre_list = [request.form['genre1'],request.form['genre2'],request.form['genre3'],request.form['genre4'],request.form['genre5']]
            session['genre_list'] = [genre for genre in genre_list if genre != 'None']

            # session['genre1'] = request.form['genre1']
            # session['genre2'] = request.form['genre2']
            # session['genre3'] = request.form['genre3']
            # session['genre4'] = request.form['genre4']
            # session['genre5'] = request.form['genre5']

            session['min_year'] = int(request.form['MinYear']) if request.form['MinYear'] else 0
            session['max_year'] = int(request.form['MaxYear']) if request.form['MaxYear'] else 9999
            session['lmr'] = float(request.form['LMR']) if request.form['LMR'] else 0.0
            session['imdb'] = float(request.form['IMDB']) if request.form['IMDB'] else 0.0
            session['min_runtime'] = int(request.form['MinRuntime']) if request.form['MinRuntime'] else 0
            session['max_runtime'] = int(request.form['MaxRuntime']) if request.form['MaxRuntime'] else 999

            session['recCount'] = 0
            session.modified = True
            return redirect('/newFilmFilterFinder')

    f = open('../HTML/lw14_feeling_picky.txt', 'r')
    page = f.read()
    f.close()
    return page.format(user=user)

@app.route('/newFilmFilterFinder', methods=["GET", "POST"])
def newFilmFilterFinder():
    '''Finding the ideal film for this user and their partner, same as the other
    film finder but adds a merged user.'''


    # The user is the sessions user.
    user = session["user"]
    movieRatingsList = pd.read_csv("movieRatingsList.csv", index_col='tconst')
    projRecList = pd.read_csv("pred_scores.csv", index_col='tconst')

    userProjRecList = projRecList[projRecList[user].notnull()].sort_values(user, ascending=False)
    userProjRecList = userProjRecList.merge(movieRatingsList, left_index=True, right_index=True, how='inner')[['primaryTitle', 'genre1', 'genre2', 'genre3', 'startYear', 'runtimeMinutes', 'averageRating']]
    userProjRecList = userProjRecList.merge(pd.DataFrame(movieRatingsList.iloc[:,10:].mean(axis=1),columns=['lmr_avg']), left_index=True, right_index=True)

    # genre_list = [session['genre1'],session['genre2'],session['genre3'],session['genre4'],session['genre5']]
    # genre_list_filtered = [genre for genre in genre_list if genre != 'None']


    if session['genre_list'] != []:
        userProjRecList = userProjRecList[
            (userProjRecList['genre1'].isin(session['genre_list'])) |
            (userProjRecList['genre2'].isin(session['genre_list'])) |
            (userProjRecList['genre2'].isin(session['genre_list']))]

    userProjRecList = userProjRecList[
        (userProjRecList['startYear'] >= session['min_year']) &
        (userProjRecList['startYear'] <= session['max_year']) &
        (userProjRecList['lmr_avg'] >= session['lmr']) &
        (userProjRecList['averageRating'] >= session['imdb']) &
        (userProjRecList['runtimeMinutes'] >= session['min_runtime']) &
        (userProjRecList['runtimeMinutes'] <= session['max_runtime'])
    ]

    if session['recCount'] >= len(userProjRecList):
                f = open('../HTML/lw18_feeling_picky_out_of_films.txt', 'r')
                page = f.read()
                f.close()
                return page.format(user=user)

    film_tconst = userProjRecList.index[session['recCount']]

    null_rating_warning = ''

    if request.method == "POST":

        if request.form["action"] == "enter rating":
            rating_value = float(request.form["amountRange"])
            if rating_value > 0 and rating_value <= 10:
                movieRatingsList.at[film_tconst, session['user']] = rating_value
                movieRatingsList.at[film_tconst, 'NoUserInput'] = False
                movieRatingsList.to_csv('movieRatingsList.csv', index=True)

                projRecList = pd.read_csv("pred_scores.csv", index_col='tconst')
                projRecList.at[film_tconst, session['user']] = np.nan
                projRecList.to_csv('pred_scores.csv', index=True)


                userProjRecList = projRecList[projRecList[user].notnull()].sort_values(user, ascending=False)
                userProjRecList = userProjRecList.merge(movieRatingsList, left_index=True, right_index=True, how='inner')[['primaryTitle', 'genre1', 'genre2', 'genre3', 'startYear', 'runtimeMinutes', 'averageRating']]
                userProjRecList = userProjRecList.merge(pd.DataFrame(movieRatingsList.iloc[:,10:].mean(axis=1),columns=['lmr_avg']), left_index=True, right_index=True)

                if session['genre_list'] != []:
                    userProjRecList = userProjRecList[
                        (userProjRecList['genre1'].isin(session['genre_list'])) |
                        (userProjRecList['genre2'].isin(session['genre_list'])) |
                        (userProjRecList['genre2'].isin(session['genre_list']))]

                userProjRecList = userProjRecList[
                    (userProjRecList['startYear'] >= session['min_year']) &
                    (userProjRecList['startYear'] <= session['max_year']) &
                    (userProjRecList['lmr_avg'] >= session['lmr']) &
                    (userProjRecList['averageRating'] >= session['imdb']) &
                    (userProjRecList['runtimeMinutes'] >= session['min_runtime']) &
                    (userProjRecList['runtimeMinutes'] <= session['max_runtime'])
                ]

                film_tconst = userProjRecList.index[session['recCount']]



            else:
                null_rating_warning = '<p class="error-text">Enter a rating greater than 0 and less than or equal to 10</p> <!-- Error message -->'

        if request.form["action"] == "next film":
            session['recCount'] += 1
            session.modified = True

            if session['recCount'] >= len(userProjRecList):
                f = open('../HTML/lw18_feeling_picky_out_of_films.txt', 'r')
                page = f.read()
                f.close()
                return page.format(user = user)
            return redirect('/newFilmFilterFinder')





    film_name = userProjRecList.loc[film_tconst, 'primaryTitle']
    film_year = userProjRecList.loc[film_tconst, 'startYear']
    film_genres = ' • '.join(userProjRecList.loc[film_tconst, ['genre1','genre2', 'genre3']].dropna().astype(str))

    try: film_desc = descFilm(str(film_name), int(film_year))
    except: film_desc = ''
    try: film_photo = filmPhoto(film_name, film_year)
    except: film_photo = ''

    # Geting streams
    try: film_streams = filmStreams(film_name, film_year, session['country'])
    except: film_streams = []

    stream_string = ''
    for stream in film_streams:
        imageLink = streamImagesLinks(stream, 'AU')
        if imageLink:
            stream_string += ''' <a href="{link}"><img src="{photo}" style="max-width: 40px; height: auto; margin-bottom: 4px;" alt="{stream}"></a>'''.format(photo=imageLink[0], stream=stream, link=imageLink[1])




    # This is what they user sees, they are recommended a film and can say yes ot no.
    f = open('../HTML/lw17_feeling_picky_rec.txt', 'r')
    page = f.read()
    f.close()
    return page.format(
        user = user,
        film_name=film_name,
        film_year=film_year,
        film_photo=film_photo,
        stream_string=stream_string,
        film_desc=film_desc,
        film_genres=film_genres,
        null_rating_warning=null_rating_warning)

@app.route('/newPartnerFind', methods=["GET", "POST"])
def newPartnerFind():
    '''Searching for the partner's account.'''

    # Film data csv file.
    user_file = pd.read_csv("accountDetails.csv")
    user_file.index = user_file['User'].str.lower()

    user = session['user']

    if 'viewing_party' not in session:
        session['viewing_party'] = [user]

    viewing_party = session['viewing_party']

    warning_string = ''



    if request.method == "POST":
        if request.form["action"] == "add_user":
            adduser = request.form['AddUser']
            if (adduser.lower() in user_file.index) and (adduser.lower() not in [u.lower() for u in viewing_party]):
                viewing_party.append(user_file.loc[adduser.lower(), 'User'])
                session['viewing_party'] = viewing_party

            elif (adduser.lower() not in user_file.index):
                warning_string = '''<p class="error-text">User not found, double check their username</p> <!-- Error message -->'''

            else:
                warning_string = '''<p class="error-text">User already in the viewing party</p> <!-- Error message -->'''


        for u in viewing_party:
            if request.form["action"] == u:
                viewing_party.remove(u)
                session['viewing_party'] = viewing_party

        if request.form["action"] == "find_a_film":
            session['recCount'] = 0
            session.modified = True
            return redirect('/newPartnerFilmFinder')


    viewing_party_string = ''

    for u in viewing_party:
        remove_button = f'''<button class="w3-bar-item w3-button w3-padding w3-khaki" id="submit" name="action" value="{u}" style="margin-top: 2px;">Remove</button>'''
        if u == user:
            remove_button = ''
        viewing_party_string += f'''
                        <div class="w3-third">
            				<div class="w3-card w3-container" style="margin: 5px; margin-bottom: 10px; min-height: 15vw;">
            					<h2><b>{u}</b></h2>
                                {remove_button}
            				</div>
            			</div>
            '''


    f = open('../HTML/lw15_feeling_friendly.txt', 'r')
    page = f.read()
    f.close()
    return page.format(
        user=user,
        warning_string=warning_string,
        viewing_party_string=viewing_party_string)


@app.route('/newPartnerFilmFinder', methods=["GET", "POST"])
def newPartnerFilmFinder():
    '''Finding the ideal film for this user and their partner, same as the other
    film finder but adds a merged user.'''

    # Reads in the film data csv.
    user = session['user']
    movieRatingsList = pd.read_csv("movieRatingsList.csv", index_col='tconst')
    projRecList = pd.read_csv("pred_scores.csv", index_col='tconst')

    viewing_party = session['viewing_party']

    projRecList = projRecList[viewing_party].dropna(how='any')

    projRecList['combined'] = projRecList.mean(axis=1)
    projRecList = projRecList.sort_values('combined', ascending=False)

    film_tconst = projRecList.index[session['recCount']]
    film_name = movieRatingsList.loc[film_tconst, 'primaryTitle']
    film_year = movieRatingsList.loc[film_tconst, 'startYear']
    film_genres = ' • '.join(movieRatingsList.loc[film_tconst, ['genre1','genre2', 'genre3']].dropna().astype(str))
    film_desc = descFilm(str(film_name), int(film_year))
    film_photo = filmPhoto(film_name, film_year)

    try: film_desc = descFilm(str(film_name), int(film_year))
    except: film_desc = ''
    try: film_photo = filmPhoto(film_name, film_year)
    except: film_photo = ''

    # Geting streams
    try: film_streams = filmStreams(film_name, film_year, session['country'])
    except: film_streams = []

    stream_string = ''
    for stream in film_streams:
        imageLink = streamImagesLinks(stream, 'AU')
        if imageLink:
            stream_string += ''' <a href="{link}"><img src="{photo}" style="max-width: 40px; height: auto; margin-bottom: 4px;" alt="{stream}"></a>'''.format(photo=imageLink[0], stream=stream, link=imageLink[1])

    if request.method == "POST":

        if request.form["action"] == "next film":
            session['recCount'] += 1
            session.modified = True
            return redirect('/newPartnerFilmFinder')


    # This is what they user sees, they are recommended a film and can say yes ot no.
    f = open('../HTML/lw16_party_film_rec.txt', 'r')
    page = f.read()
    f.close()
    return page.format(
        user = user,
        film_name=film_name,
        film_year=film_year,
        film_photo=film_photo,
        stream_string=stream_string,
        film_desc=film_desc,
        film_genres=film_genres)

@app.route('/feelingPretentious', methods=["GET", "POST"])
def feelingPretentious():
    '''Finding the ideal unrated film for this user, same as the other
    film finder but the film hasn't been rated by any other user.'''

    # The user is the sessions user.
    user = session["user"]
    movieRatingsList = pd.read_csv("movieRatingsList.csv", index_col='tconst')
    projRecList = pd.read_csv("fp_pred_scores.csv", index_col='tconst')

    userProjRecList = projRecList[projRecList[user].notnull()].sort_values(user, ascending=False)
    userProjRecList = userProjRecList.merge(movieRatingsList, left_index=True, right_index=True, how='inner')[['primaryTitle', 'genre1', 'genre2', 'genre3', 'startYear', 'runtimeMinutes', 'averageRating']]
    userProjRecList = userProjRecList.merge(pd.DataFrame(movieRatingsList.iloc[:,10:].mean(axis=1),columns=['lmr_avg']), left_index=True, right_index=True)

    if session['recCount'] >= len(userProjRecList):
                f = open('../HTML/lw18_feeling_picky_out_of_films.txt', 'r')
                page = f.read()
                f.close()
                return page.format(user=user)

    film_tconst = userProjRecList.index[session['recCount']]

    null_rating_warning = ''

    if request.method == "POST":

        if request.form["action"] == "enter rating":
            rating_value = float(request.form["amountRange"])
            if rating_value > 0 and rating_value <= 10:
                movieRatingsList.at[film_tconst, session['user']] = rating_value
                movieRatingsList.at[film_tconst, 'NoUserInput'] = False
                movieRatingsList.to_csv('movieRatingsList.csv', index=True)

                projRecList = pd.read_csv("fp_pred_scores.csv", index_col='tconst')
                projRecList.drop(film_tconst, axis=0, inplace=True)
                projRecList.to_csv('fp_pred_scores.csv', index=True)


                userProjRecList = projRecList[projRecList[user].notnull()].sort_values(user, ascending=False)
                userProjRecList = userProjRecList.merge(movieRatingsList, left_index=True, right_index=True, how='inner')[['primaryTitle', 'genre1', 'genre2', 'genre3', 'startYear', 'runtimeMinutes', 'averageRating']]
                userProjRecList = userProjRecList.merge(pd.DataFrame(movieRatingsList.iloc[:,10:].mean(axis=1),columns=['lmr_avg']), left_index=True, right_index=True)

                film_tconst = userProjRecList.index[session['recCount']]

            else:
                null_rating_warning = '<p class="error-text">Enter a rating greater than 0 and less than or equal to 10</p> <!-- Error message -->'

        if request.form["action"] == "next film":
            session['recCount'] += 1
            session.modified = True

            if session['recCount'] >= len(userProjRecList):
                f = open('../HTML/lw18_feeling_picky_out_of_films.txt', 'r')
                page = f.read()
                f.close()
                return page.format(user = user)
            return redirect('/feelingPretentious')

    film_name = userProjRecList.loc[film_tconst, 'primaryTitle']
    film_year = userProjRecList.loc[film_tconst, 'startYear']
    film_genres = ' • '.join(userProjRecList.loc[film_tconst, ['genre1','genre2', 'genre3']].dropna().astype(str))

    try: film_desc = descFilm(str(film_name), int(film_year))
    except: film_desc = ''
    try: film_photo = filmPhoto(film_name, film_year)
    except: film_photo = ''

    # Geting streams
    try: film_streams = filmStreams(film_name, film_year, session['country'])
    except: film_streams = []

    stream_string = ''
    for stream in film_streams:
        imageLink = streamImagesLinks(stream, 'AU')
        if imageLink:
            stream_string += ''' <a href="{link}"><img src="{photo}" style="max-width: 40px; height: auto; margin-bottom: 4px;" alt="{stream}"></a>'''.format(photo=imageLink[0], stream=stream, link=imageLink[1])

    # This is what they user sees, they are recommended a film and can say yes ot no.
    f = open('../HTML/lw19_feeling_pretentious_rec.txt', 'r')
    page = f.read()
    f.close()
    return page.format(
        user = user,
        film_name=film_name,
        film_year=film_year,
        film_photo=film_photo,
        stream_string=stream_string,
        film_desc=film_desc,
        film_genres=film_genres,
        null_rating_warning=null_rating_warning)

@app.route('/lboxUpload', methods=["GET", "POST"])
def lboxUpload():
    '''Finding the ideal unrated film for this user, same as the other
    film finder but the film hasn't been rated by any other user.'''

    user = session["user"]
    warning = ''

    uploaded_file = request.files.get('zip_file')
    if not uploaded_file:
        warning = ''

    elif not uploaded_file.filename.endswith('.zip'):
        warning = '<p class="error-text">Invalid file format. Please upload the .zip file directly from Letterboxd.</p> <!-- Error message -->'

    else:
        # Read the zip file in memory
        zip_bytes = io.BytesIO(uploaded_file.read())
        try:
            with zipfile.ZipFile(zip_bytes) as zip_ref:
                # List all files inside the zip
                file_list = zip_ref.namelist()

                # Check if ratings.csv is present
                if 'ratings.csv' not in file_list:
                    warning = '<p class="error-text">Make sure the file is the zip file from Letterboxd.</p> <!-- Error message -->'

                # Read the ratings.csv file into a DataFrame
                with zip_ref.open('ratings.csv') as ratings_file:
                    lbox_df = pd.read_csv(ratings_file)

                lbox_df = lbox_df[['Name', 'Year', 'Rating']]

                movieRatingsList = pd.read_csv("movieRatingsList.csv")

                movieRatingsList = movieRatingsList.merge(lbox_df, how='left', left_on=['primaryTitle', 'startYear'], right_on=['Name', 'Year'])
                movieRatingsList[user] = movieRatingsList[user].fillna(movieRatingsList['Rating'] * 2)
                movieRatingsList.loc[movieRatingsList['NoUserInput'] & (movieRatingsList['Rating'].notnull()), 'NoUserInput'] = False
                movieRatingsList.drop(['Name', 'Year', 'Rating'], axis=1, inplace=True)

                movieRatingsList.to_csv('movieRatingsList.csv', index=False)

                warning = '<p class="w3-text-green w3-large"><strong>Upload successful!</strong> Your Letterboxd ratings have been merged with your LMR ratings.</p>'

        except zipfile.BadZipFile:
            warning = '<p class="error-text">Make sure the file is the zip file from Letterboxd.</p> <!-- Error message -->'

    # This is what they user sees, they are recommended a film and can say yes ot no.
    f = open('../HTML/lw20_lbox_upload.txt', 'r')
    page = f.read()
    f.close()
    return page.format(
        user=user,
        warning=warning
        )


if __name__ == "__main__":
    app.run(debug=True)

