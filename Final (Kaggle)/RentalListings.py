import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

colors = sns.color_palette()

trainData = pd.read_json("train.json")
testData = pd.read_json("test.json")

## Initialize empty dictionaries
pValues = {'listing_id': {}, 'lowPrice': {}, 'mediumPrice': {}, 'highPrice': {},}
# Variables for dataframes
trainLowInterestDistribution = {}
trainMedInterestDistribution = {}
trainHighInterestDistribution = {}

# Sample distributions to insert into dataframes
trainLowInterestNumFeatureList = {}
trainMedInterestNumFeatureList = {}
trainHighInterestNumFeatureList = {}
trainLowInterestNumPhotosList = {}
trainMedInterestNumPhotosList = {}
trainHighInterestNumPhotosList = {}
trainLowInterestListingIdList = {}
trainMedInterestListingIdList = {}
trainHighInterestListingIdList = {}
trainLowInterestPriceList = {}
trainMedInterestPriceList = {}
trainHighInterestPriceList = {}

## Create the sample distribution of different interest_level rental prices
# Create the list of index to prices and number of features
for index,row in trainData.iterrows():
    interestLevel = row['interest_level']
    if interestLevel == 'low':
        trainLowInterestPriceList[index] = row['price']
        trainLowInterestNumFeatureList[index] = len(row['features'])
        trainLowInterestNumPhotosList[index] = len(row['photos'])
        trainLowInterestListingIdList[index] = row['listing_id']
    elif interestLevel == 'medium':
        trainMedInterestPriceList[index] = row['price']
        trainMedInterestNumFeatureList[index] = len(row['features'])
        trainMedInterestNumPhotosList[index] = len(row['photos'])
        trainMedInterestListingIdList[index] = row['listing_id']
    elif interestLevel == 'high':
        trainHighInterestPriceList[index] = row['price']
        trainHighInterestNumFeatureList[index] = len(row['features'])
        trainHighInterestNumPhotosList[index] = len(row['photos'])
        trainHighInterestListingIdList[index] = row['listing_id']

# Convert dictionaries to Series
trainMedInterestPriceList = pd.Series(trainMedInterestPriceList)
trainLowInterestPriceList = pd.Series(trainLowInterestPriceList)
trainHighInterestPriceList = pd.Series(trainHighInterestPriceList)
trainLowInterestNumPhotosList = pd.Series(trainLowInterestNumPhotosList)
trainMedInterestNumPhotosList = pd.Series(trainMedInterestNumPhotosList)
trainHighInterestNumPhotosList = pd.Series(trainHighInterestNumPhotosList)
trainLowInterestNumFeatureList = pd.Series(trainLowInterestNumFeatureList)
trainMedInterestNumFeatureList = pd.Series(trainMedInterestNumFeatureList)
trainHighInterestNumFeatureList = pd.Series(trainHighInterestNumFeatureList)
trainLowInterestListingIdList = pd.Series(trainLowInterestListingIdList)
trainMedInterestListingIdList = pd.Series(trainMedInterestListingIdList)
trainHighInterestListingIdList = pd.Series(trainHighInterestListingIdList)

# Create the dataframes of interest_level distributions of prices
## We will use these distributions to generate the pValues
d = {
    'price': trainLowInterestPriceList,
    'photos': trainLowInterestNumPhotosList,
    'features': trainLowInterestNumFeatureList,
    'listing_id': trainLowInterestListingIdList,
}
trainLowInterestDistribution = pd.DataFrame(d)
d = {
    'price': trainMedInterestPriceList,
    'photos': trainMedInterestNumPhotosList,
    'features': trainMedInterestNumFeatureList,
    'listing_id': trainMedInterestListingIdList,
}
trainMedInterestDistribution = pd.DataFrame(d)
d = {
    'price': trainHighInterestPriceList,
    'photos': trainHighInterestNumPhotosList,
    'features': trainHighInterestNumFeatureList,
    'listing_id': trainHighInterestListingIdList,
}
trainHighInterestDistribution = pd.DataFrame(d)

# Demonstration of what was done above
print("Low Interest Rentals:\n", trainLowInterestDistribution.head())
print("Medium Interest Rentals:\n", trainMedInterestDistribution.head())
print("High Interest Rentals:\n", trainHighInterestDistribution.head())

# Now, we generate pValues by calling stats.percentileofscore and doing a two-tailed result
for index, row in trainData.iterrows():
    pValLow = 0
    pValMed = 0
    pValHigh = 0
    pValues['listing_id'][index] = row['listing_id']
    percentileLow = stats.percentileofscore(trainLowInterestDistribution['price'], row['price'])
    percentileMed = stats.percentileofscore(trainMedInterestDistribution['price'], row['price'])
    percentileHigh = stats.percentileofscore(trainHighInterestDistribution['price'], row['price'])

    # This step is finding the pValue using the quantile of the distribution
    if percentileLow <= 50:
        pValLow = (percentileLow*2)/100
    elif percentileLow > 50:
        pValLow = ((100 - percentileLow)*2)/100
    pValues['lowPrice'][index] = pValLow

    if percentileMed <= 50:
        pValMed = (percentileMed*2)/100
    elif percentileMed > 50:
        pValMed = ((100 - percentileMed)*2)/100
    pValues['mediumPrice'][index] = pValMed

    if percentileHigh <= 50:
        pValHigh = (percentileHigh*2)/100
    elif percentileHigh > 50:
        pValHigh = ((100 - percentileHigh)*2)/100
    pValues['highPrice'][index] = pValHigh

pValues = pd.DataFrame(pValues)
print(pValues.head())

# Then, we do this whole process for every factor that we deem influential to interest_level.
# After that, we take those pVals and average them for each listing, and then normalize them to form the
# probabilities.