import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

trainData = pd.read_json("train.json")
#testData = pd.read_json("test.json")

## Initialize empty dictionaries
pValues = {'listing_id': {},}
probabilityDataFrame = {'low': {}, 'medium': {}, 'high': {}, 'listing_id': {},}
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

# To create a fully iterable list of counts
concatList = pd.concat([trainLowInterestDistribution, trainMedInterestDistribution, trainHighInterestDistribution])
concatList = concatList.sort_index(0)

# Generate the dictionaries for the pValues of each respective category
for column, items in concatList.iteritems():
    if column == 'listing_id':
        continue
    pValues[column+'_low'] = {}
    pValues[column+'_medium'] = {}
    pValues[column+'_high'] = {}


DELETEME = 0
# Now, we generate pValues by calling stats.percentileofscore and doing a two-tailed result
# Then we store the pValues into the pValue dictionaries we just created
for index, row in concatList.iterrows():
    if DELETEME == 10:
        break
    pValCounter = 0
    pValLowSum = 0
    pValMedSum = 0
    pValHighSum = 0
    for column, items in concatList.iteritems():
        if column == 'listing_id':
            continue
        pValCounter += 1
        pValLow = 0
        pValMed = 0
        pValHigh = 0
        pValues['listing_id'][index] = row['listing_id']
        percentileLow = stats.percentileofscore(trainLowInterestDistribution[column], row[column])
        percentileMed = stats.percentileofscore(trainMedInterestDistribution[column], row[column])
        percentileHigh = stats.percentileofscore(trainHighInterestDistribution[column], row[column])

        # This step is finding the pValue using the quantile of the distribution
        if percentileLow <= 50:
            pValLow = (percentileLow*2)/100
        elif percentileLow > 50:
            pValLow = ((100 - percentileLow)*2)/100
        pValues[column+'_low'][index] = pValLow

        if percentileMed <= 50:
            pValMed = (percentileMed*2)/100
        elif percentileMed > 50:
            pValMed = ((100 - percentileMed)*2)/100
        pValues[column+'_medium'][index] = pValMed

        if percentileHigh <= 50:
            pValHigh = (percentileHigh*2)/100
        elif percentileHigh > 50:
            pValHigh = ((100 - percentileHigh)*2)/100
        pValues[column+'_high'][index] = pValHigh

        # Algorithm for probabilities of interest takes the average of the pValues as a sort of hypothesis test
        pValLowSum += pValLow
        pValMedSum += pValMed
        pValHighSum += pValHigh

    # The average for each category pValue is taken, and then all are normalized to sum to one
    pValLowAverage = pValLowSum / pValCounter
    pValMedAverage = pValMedSum /pValCounter
    pValHighAverage = pValHighSum / pValCounter
    pValSum = pValLowAverage + pValMedAverage + pValHighAverage

    # Forming the final probabilities based on the pValues
    lowProbability = pValLowAverage/pValSum
    medProbability = pValMedAverage/pValSum
    highProbability = pValHighAverage/pValSum

    probabilityDataFrame['low'][index] = lowProbability
    probabilityDataFrame['medium'][index] = medProbability
    probabilityDataFrame['high'][index] = highProbability
    probabilityDataFrame['listing_id'][index] = row['listing_id']

    DELETEME += 1

pValues = pd.DataFrame(pValues)
probabilityDataFrame = pd.DataFrame(probabilityDataFrame)

# Arrange order of columns for CSV file
cols = probabilityDataFrame.columns.tolist()
reorderedCols =[1,0,3,2]
cols = [ cols[i] for i in reorderedCols]
probabilityDataFrame = probabilityDataFrame[cols]
probabilityDataFrame.to_csv("trainListings.csv", index=False)

print("P-Values:\n", pValues.head())
print("Probabilities:\n", probabilityDataFrame.head())
# Now, we take those pVals and average the resoective categories for each listing, and then normalize them to form the
# probabilities.