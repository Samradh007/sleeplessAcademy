#----------------------------------------------------------------------------#
# Imports
#----------------------------------------------------------------------------#

from flask import Flask, render_template, request, Response,jsonify, session
# from flask.ext.sqlalchemy import SQLAlchemy
import logging
from logging import Formatter, FileHandler
from forms import *
import os
from tools import is_url_ok
from model import DrowsyDetector
import cv2

#----------------------------------------------------------------------------#
# App Config.
#----------------------------------------------------------------------------#

app = Flask(__name__)
app.config.from_object('config')
pyModel = DrowsyDetector()
#db = SQLAlchemy(app)

# Automatically tear down SQLAlchemy.
'''
@app.teardown_request
def shutdown_session(exception=None):
    db_session.remove()
'''

# Login required decorator.
'''
def login_required(test):
    @wraps(test)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return test(*args, **kwargs)
        else:
            flash('You need to login first.')
            return redirect(url_for('login'))
    return wrap
'''
#----------------------------------------------------------------------------#
# Controllers.
#----------------------------------------------------------------------------#


@app.route('/')
def home():
    return render_template('pages/home.html')

@app.route('/courses')
def courses():
    return render_template('pages/courses.html')

@app.route('/watch/<title>/<id>')
def watch(title, id):
        youtube_url = 'https://www.youtube.com/watch?v={video_id}'.format(video_id=id) 
        if False == is_url_ok(youtube_url) :
            url = 'https://www.youtube.com/watch?v={video_id}'.format(video_id='dQw4w9WgXcQ')
            return render_template('pages/watchScreen.html', video_title = 'Nothing to see here!', yt_url=url), 404
        else:        
            return render_template('pages/watchScreen.html', video_title = title, yt_url=youtube_url, yt_id=id)


@app.route('/get_label')    
def get_label():
    model_type = request.args.get('modelType', 'default_if_none')
    input_arr = createInputArray(request.args.to_dict(flat=False))
    output = {}
    label = startModel(input_arr, model_type)
    output['label'] = label
    return jsonify(output)

def startModel(input_arr, model_type):
    return pyModel.get_classification(input_arr, model_type)

def createInputArray(multiDict):
    input_arr = []
    i = 0
    y = []
    multiDict.pop('modelType')
    for key, value in multiDict.items():
        input_arr.append(value)
    return input_arr


@app.route('/about')
def about():
    return render_template('pages/about.html')

@app.route('/login')
def login():
    form = LoginForm(request.form)
    return render_template('forms/login.html', form=form)

@app.route('/register')
def register():
    form = RegisterForm(request.form)
    return render_template('forms/register.html', form=form)


@app.route('/forgot')
def forgot():
    form = ForgotForm(request.form)
    return render_template('forms/forgot.html', form=form)

# Error handlers.


@app.errorhandler(500)
def internal_error(error):
    #db_session.rollback()
    return render_template('errors/500.html'), 500


@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

if not app.debug:
    file_handler = FileHandler('error.log')
    file_handler.setFormatter(
        Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    )
    app.logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.info('errors')

#----------------------------------------------------------------------------#
# Launch.
#----------------------------------------------------------------------------#

# Default port:
if __name__ == '__main__':
    app.run()

# Or specify port manually:
""" 
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 6000))
    app.run(host='0.0.0.0', port=port)

 """