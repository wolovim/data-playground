from flask import Flask, render_template
import pandas as pd
import seaborn as sns


app = Flask(__name__)
titanic_df = pd.read_csv('./data/titanic-training-set.csv')

def build_titanic_genders():
  sns.set(font="serif")
  plot = sns.factorplot('Sex', data=titanic_df, kind='count')
  plot.fig.savefig("static/titanic/gender.png")

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/titanic')
def titanic():
  build_titanic_genders()
  return render_template('titanic.html', df=titanic_df)

if __name__ == '__main__':
  app.run()
