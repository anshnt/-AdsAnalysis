from flask import Flask
import sys
import os
ROOT_DIR = os.path.abspath(os.curdir)
sys.path.insert(0, ROOT_DIR+'/models')
from user_behavior import *
from ecom_behaviour import *
import json

app = Flask(__name__)



userBehavior = '/user_behavior'

@app.route("/")
def home():
  return 'home'

@app.route(userBehavior+"/getPrecision")
def gettingPrecision():
  return precision()

@app.route(userBehavior+"/getPredictions")
def gettingPredictions():
  data = getPredictions()
  return json.dumps(data.tolist())

@app.route(userBehavior+"/getAgeDailyTimeSpentonSite")
def gettingAgeDailyTimeSpentonSite():
  return getAgeDailyTimeSpentonSite()

@app.route(userBehavior+"/all")
def gettingAllData():
  return all()

@app.route(userBehavior+"/getAgeAreaIncome")
def gettingAgeAreaIncome():
  return getAgeAreaIncome()

if __name__ == "__main__":
  app.run(debug = True)

  # app.run(host="0.0.0.0", port=8080, debug=True)

# # flask depends on this env variable to find the main file
# export FLASK_APP=user-behavior-advertisement.py

# # now we just need to ask flask to run
# flask run

# # * Serving Flask app "hello"
# # * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

# set FLASK_APP=app.py
# $env:FLASK_APP = "app.py"
# python -m  flask run