
import pandas as pd
from pomegranate import DiscreteDistribution,ConditionalProbabilityTable,State,BayesianNetwork

df = pd.read_excel(r"C:\Users\Sai Kiran\Desktop\India_Test_Stats.xlsx", encoding='UTF-8')

def returnPriorProbability(df, column):
    priorProbDict = {}
    for j in df[column].unique():
        priorProbDict[j] = df[df[column] == j].count().iloc[0] / df[[column]].count().iloc[0]
        priorProbDict[j] = df[df[column] == j].count().iloc[0] / df[[column]].count().iloc[0]
    return priorProbDict

def returnConditionalProbability(df, Location, AshwinPlaying):
    condProbDict = []
    priorProb = returnPriorProbability(df, Location)
    for j in df[Location].unique():
        for k in df[AshwinPlaying].unique():
            val = df[(df[Location] == j) & (df[AshwinPlaying]  == k)].count().iloc[0] / df[[Location]].count().iloc[0]/priorProb[j]
            
            condProbDict.append([j,k,val])
    return condProbDict

arr = returnConditionalProbability(df, 'Location', 'Ashwin')
arr
arr = returnConditionalProbability(df, 'Toss', 'Bat')
arr

arr = returnConditionalProbability(df,'Bat', 'Result')
arr

location = DiscreteDistribution(returnPriorProbability(df, 'Location'))
toss = DiscreteDistribution(returnPriorProbability(df, 'Toss'))

ashwin = ConditionalProbabilityTable(returnConditionalProbability(df, 'Location', 'Ashwin'),[location])
batting = ConditionalProbabilityTable(returnConditionalProbability(df, 'Toss', 'Bat'), [toss])
result = ConditionalProbabilityTable(returnConditionalProbability(df, 'Bat', 'Result'), [batting])


sLocation = State(location, name="Location")
sToss = State(toss, name="Toss")
sBatting = State(batting, name="Batting")
sAshwin = State(ashwin, name="Ashwin")
sResult = State(result, name="Result")


# Create the Bayesian network object with a useful name
model = BayesianNetwork("Ashwin Playing Problem")

# Add the three states to the network 
model.add_states(sLocation, sToss, sBatting, sAshwin, sResult)
model.add_edge(sLocation, sAshwin)
model.add_edge(sToss, sBatting)
model.add_edge(sBatting, sResult)
model.bake()



model.predict_proba([None, None,'2nd', 'Y', 'won'])[1]
model.predict_proba([None, None,'2nd', 'N', 'won'])[0]

model.predict_proba([None, None,'2nd', 'Y', 'lost'])[1]

model.predict_proba([None, None,'2nd', 'N', 'lost'])[0]
