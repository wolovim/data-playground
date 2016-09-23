from flask import Flask, render_template
import pandas as pd
from pandas import Series, DataFrame

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/titanic')
def titanic():
  titanic_df = pd.read_csv('./data/titanic-training-set.csv')
  return render_template('titanic.html', df=titanic_df)

if __name__ == '__main__':
  app.run()
