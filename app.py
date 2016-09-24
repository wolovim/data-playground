from flask import Flask, render_template
import pandas as pd
import seaborn as sns


app = Flask(__name__)
titanic_df = pd.read_csv('./data/titanic-training-set.csv')

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/titanic')
def titanic():
  build_titanic_demographics()
  return render_template('titanic.html', df=titanic_df)

def build_titanic_demographics():
  # Passengers by gender
  plot = sns.factorplot('Sex', data=titanic_df, kind='count')
  plot.fig.savefig('static/titanic/gender.png')

  # Classes by gender
  plot_2 = sns.factorplot('Pclass', data=titanic_df, hue='Sex', kind='count', legend_out=False)
  plot_2.fig.savefig('static/titanic/class.png')

  # Add a column to identify male, female, or child
  titanic_df['person'] = titanic_df[['Age', 'Sex']].apply(identify_children, axis=1)
  plot_3 = sns.factorplot('Pclass', data=titanic_df, hue='person', kind='count', legend_out=False)
  plot_3.fig.savefig('static/titanic/class-with-child.png')

def identify_children(passenger):
  age,sex = passenger
  return 'child' if age < 16 else sex

if __name__ == '__main__':
  app.run()
