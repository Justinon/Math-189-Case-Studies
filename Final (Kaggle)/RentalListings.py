import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

colors = sns.color_palette()

trainData = pd.read_json("train.json")
testData = pd.read_json("test.json")

# Initialize empty dictionaries
pricePvalues = {}
trainLowInterestIndexToPrice = {}
trainMedInterestIndexToPrice = {}
trainHighInterestIndexToPrice = {}
trainLowInterestList = {}
trainMedInterestList = {}
trainHighInterestList = {}

## Create the sample distribution of different interest_level rental prices
# Create the list of index to prices
for index,row in trainData.iterrows():
    interestLevel = row['interest_level']
    if interestLevel == 'low':
        trainLowInterestList[index] = row['price']
    elif interestLevel == 'medium':
        trainMedInterestList[index] = row['price']
    elif interestLevel == 'high':
        trainHighInterestList[index] = row['price']

trainMedInterestList = pd.Series(trainMedInterestList)
trainLowInterestList = pd.Series(trainLowInterestList)
trainHighInterestList = pd.Series(trainHighInterestList)

# Create the dataframes of interest_level distributions of prices
## We will use these distributions to generate the pValues
d = {'price': trainLowInterestList,}
trainLowInterestIndexToPrice = pd.DataFrame(d)
d = {'price': trainMedInterestList,}
trainMedInterestIndexToPrice = pd.DataFrame(d)
d = {'price': trainHighInterestList,}
trainHighInterestIndexToPrice = pd.DataFrame(d)

# Now, we generate pValues by calling stats.percentileofscore(distribution, score) and doing a two-tailed result

# Then, we do this whole process for every factor that we deem influential to interest_level.
# After that, we take those pVals and average them for each listing, and then normalize them to form the
# probabilities.