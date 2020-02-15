
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.stats import poisson,skellam

import statsmodels.api as sm
import statsmodels.formula.api as smf

dataset = pd.read_csv('campeonato-brasileiro-full.csv')
dataset.head()



dataset = dataset[['Clube 1', 'Clube 2', 'p1', 'p2']]

dataset = dataset.rename(columns = {'Clube 1': 'TimeCasa', 'Clube 2':'TimeVisitante'})
dataset = dataset.rename(columns = {'p1': 'GolsCasa', 'p2':'GolsVisitante'})

dataset['TimeCasa'] = dataset['TimeCasa'].str.lower()
dataset['TimeVisitante'] = dataset['TimeVisitante'].str.lower()

goal_model_data = pd.concat([dataset[['TimeCasa','TimeVisitante','GolsCasa']].assign(home=0).rename(
            columns={'TimeCasa':'team', 'TimeVisitante':'opponent','GolsCasa':'goals'}),
           dataset[['TimeVisitante','TimeCasa','GolsVisitante']].assign(home=0).rename(
            columns={'TimeVisitante':'team', 'TimeCasa':'opponent','GolsVisitantes':'goals'})], sort = True)


poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data, 
                        family=sm.families.Poisson()).fit()
#poisson_model.summary()


def simulate_match(foot_model, homeTeam, awayTeam, max_goals=10):
    home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': homeTeam, 
                                                            'opponent': awayTeam,'home':12},
                                                      index=[1])).values[0]
    away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': awayTeam, 
                                                            'opponent': homeTeam,'home':0},
                                                      index=[1])).values[0]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
    
    return(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))


time_casa = 'flamengo'
time_visitante = 'atlético-pr'

fla_atpr = simulate_match(poisson_model, time_casa, time_visitante, max_goals=5)
maior = max([valor for linha in fla_atpr for valor in linha])

for i,j in enumerate(fla_atpr):
   for k,l in enumerate(j):
     if l==maior:
         print (time_casa , i, ' x ', k, time_visitante)


