from flask import Flask, render_template
import pandas as pd
from pandas import DataFrame
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
  titanic_df['Person'] = titanic_df[['Age', 'Sex']].apply(identify_children, axis=1)
  plot_3 = sns.factorplot('Pclass', data=titanic_df, hue='Person', kind='count', legend_out=False)
  plot_3.fig.savefig('static/titanic/class-with-child.png')

  # FacetGrid for age distribution by gender
  plot_4 = sns.FacetGrid(titanic_df, hue="Person",aspect=4)
  plot_4.map(sns.kdeplot, 'Age', shade=True)
  oldest = titanic_df['Age'].max()
  plot_4.set(xlim=(0,oldest))
  plot_4.add_legend()
  plot_4.fig.savefig('static/titanic/age-dist-gender.png')

  # Age distributions by class
  plot_5 = sns.FacetGrid(titanic_df, hue="Pclass",aspect=4)
  plot_5.map(sns.kdeplot, 'Age', shade=True)
  oldest = titanic_df['Age'].max()
  plot_5.set(xlim=(0,oldest))
  plot_5.add_legend()
  plot_5.fig.savefig('static/titanic/age-dist-class.png')

  # Distribution by deck
  deck = titanic_df['Cabin'].dropna()
  levels = []
  for level in deck:
    levels.append(level[0])
  cabin_df = DataFrame(levels)
  cabin_df.columns = ['Cabin']
  generations = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
  plot_6 = sns.factorplot('Cabin', data=cabin_df, x_order=generations, kind='count', palette='winter_d')
  plot_6.fig.savefig('static/titanic/deck-dist.png')

  # Passengers with family
  titanic_df['Alone'] = titanic_df.Parch + titanic_df.SibSp
  titanic_df['Alone'].loc[titanic_df['Alone'] > 0] = 'Family'
  titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'
  plot_7 = sns.factorplot('Alone', data=titanic_df, kind='count')
  plot_7.fig.savefig('static/titanic/family.png')

  # Survival rate
  titanic_df['Survivor'] = titanic_df.Survived.map({0: "no", 1: "yes"})
  plot_8 = sns.factorplot('Survivor', data=titanic_df, kind='count')
  plot_8.fig.savefig('static/titanic/survival-rate.png')

  # Survival rate: class
  plot_9 = sns.factorplot('Pclass', 'Survived', data=titanic_df)
  plot_9.fig.savefig('static/titanic/survival-class.png')

  # Survival rate: class by gender
  plot_10 = sns.factorplot('Pclass', 'Survived', data=titanic_df, hue='Person', legend_out=False)
  plot_10.fig.savefig('static/titanic/survival-class-gender.png')

  # Survival rate: age
  plot_11 = sns.lmplot('Age', 'Survived', data=titanic_df)
  plot_11.fig.savefig('static/titanic/survival-age.png')

  # Survival rate: age by class
  x_bins = [10, 20, 40, 60, 80]
  plot_12 = sns.lmplot('Age', 'Survived', data=titanic_df, hue='Pclass', palette='winter', x_bins=x_bins, legend_out=False)
  plot_12.fig.savefig('static/titanic/survival-age-class.png')

  # Survival rate: age by class
  plot_13 = sns.lmplot('Age', 'Survived', data=titanic_df, hue='Sex', palette='winter', x_bins=x_bins, legend_out=False)
  plot_13.fig.savefig('static/titanic/survival-age-gender.png')

  # Survival rate: family
  plot_14 = sns.factorplot('Alone', 'Survived', data=titanic_df)
  plot_14.fig.savefig('static/titanic/survival-family.png')

  # Survival rate: deck
  titanic_df['Deck'] = titanic_df.Cabin.str[0]
  plot_15 = sns.factorplot('Deck', 'Survived', data=titanic_df, x_order=generations, join=False)
  plot_15.fig.savefig('static/titanic/survival-deck.png')

def identify_children(passenger):
  age,sex = passenger
  return 'child' if age < 16 else sex

if __name__ == '__main__':
  app.run()
