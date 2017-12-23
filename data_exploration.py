# Loading libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm

# loading data
train = pd.read_csv("/home/mota/Desktop/HousePricesML/data/train.csv")
test = pd.read_csv("/home/mota/Desktop/HousePricesML/data/test.csv")

# check missing values
train.columns[train.isnull().any()]

['LotFrontage',
 'Alley',
 'MasVnrType',
 'MasVnrArea',
 'BsmtQual',
 'BsmtCond',
 'BsmtExposure',
 'BsmtFinType1',
 'BsmtFinType2',
 'Electrical',
 'FireplaceQu',
 'GarageType',
 'GarageYrBlt',
 'GarageFinish',
 'GarageQual',
 'GarageCond',
 'PoolQC',
 'Fence',
 'MiscFeature']

# missing value counts in each of these columns
miss = train.isnull().sum() / len(train)
miss = miss[miss > 0]
miss.sort_values(inplace=True)
# print miss

# visualising missing values
miss = miss.to_frame()
miss.columns = ['count']
miss.index.names = ['Name']
miss['Name'] = miss.index

# plot the missing value count
sns.set(style="whitegrid", color_codes=True)
sns.barplot(x='Name', y='count', data=miss)
plt.xticks(rotation=90)
# plt.show()

# SalePrice
sns.distplot(train['OverallQual'])
# plt.show()

# skewness
# print "The skewness of SalePrice is {}".format(train['SalePrice'].skew())

# now transforming the target variable
target = np.log(train['SalePrice'])
#print ('Skewness is', target.skew())
sns.distplot(target)
# plt.show()

# separate variables into new data frames
numeric_data = train.select_dtypes(include=[np.number])
cat_data = train.select_dtypes(exclude=[np.number])

# delete id variable
del numeric_data['Id']

#print ("There are {} numeric and {} categorical columns in train data".format(numeric_data.shape[1],cat_data.shape[1]))

# correlation plot
corr = numeric_data.corr()
sns.heatmap(corr)
# plt.show()

# print (corr['SalePrice'].sort_values(ascending=False)[:15], '\n') #top 15 values
#print ('----------------------')
# print (corr['SalePrice'].sort_values(ascending=False)[-5:]) #last 5 values

# let's check the mean price per quality and plot it.
pivot = train.pivot_table(
    index='OverallQual', values='SalePrice', aggfunc=np.median)
pivot.sort_values
# print pivot

pivot.plot(kind='bar', color='red')
# plt.show()

# GrLivArea variable
sns.jointplot(x=train['GrLivArea'], y=train['SalePrice'])
# plt.show()

# print cat_data.describe()

sp_pivot = train.pivot_table(
    index='SaleCondition', values='SalePrice', aggfunc=np.median)
# print sp_pivot

sp_pivot.plot(kind='bar', color='red')
# plt.show()

cat = [f for f in train.columns if train.dtypes[f] == 'object']


def anova(frame):
    anv = pd.DataFrame()
    anv['features'] = cat
    pvals = []
    for c in cat:
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c] == cls]['SalePrice'].values
            samples.append(s)
        pval = stats.f_oneway(*samples)[1]
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')


cat_data['SalePrice'] = train.SalePrice.values
k = anova(cat_data)
k['disparity'] = np.log(1. / k['pval'].values)
sns.barplot(data=k, x='features', y='disparity')
plt.xticks(rotation=90)
# plt.show()

# create numeric plots
num = [f for f in train.columns if train.dtypes[f] != 'object']
num.remove('Id')
nd = pd.melt(train, value_vars=num)
n1 = sns.FacetGrid(nd, col='variable', col_wrap=4, sharex=False, sharey=False)
n1 = n1.map(sns.distplot, 'value')
# plt.show()


def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x = plt.xticks(rotation=90)


cat = [f for f in train.columns if train.dtypes[f] == 'object']

p = pd.melt(train, id_vars='SalePrice', value_vars=cat)
g = sns.FacetGrid(p, col='variable', col_wrap=2,
                  sharex=False, sharey=False, size=5)
g = g.map(boxplot, 'value', 'SalePrice')
# plt.show()

# removing outliers
train.drop(train[train['GrLivArea'] > 4000].index, inplace=True)
train.shape  # removed 4 rows`
(1456, 81)

# imputing using mode
test.loc[666, 'GarageQual'] = "TA"  # stats.mode(test['GarageQual']).mode
test.loc[666, 'GarageCond'] = "TA"  # stats.mode(test['GarageCond']).mode
test.loc[666, 'GarageFinish'] = "Unf"  # stats.mode(test['GarageFinish']).mode
test.loc[666, 'GarageYrBlt'] = "1980"  # np.nanmedian(test['GarageYrBlt'])`

# mark as missing
test.loc[1116, 'GarageType'] = np.nan

# importing function
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


def factorize(data, var, fill_na=None):
    if fill_na is not None:
        data[var].fillna(fill_na, inplace=True)
    le.fit(data[var])
    data[var] = le.transform(data[var])
    return data


# combine the data set
alldata = train.append(test)
alldata.shape
(2915, 81)

# impute lotfrontage by median of neighborhood
lot_frontage_by_neighborhood = train['LotFrontage'].groupby(
    train['Neighborhood'])

for key, group in lot_frontage_by_neighborhood:
    idx = (alldata['Neighborhood'] == key) & (alldata['LotFrontage'].isnull())
    alldata.loc[idx, 'LotFrontage'] = group.median()

# imputing missing values
alldata["MasVnrArea"].fillna(0, inplace=True)
alldata["BsmtFinSF1"].fillna(0, inplace=True)
alldata["BsmtFinSF2"].fillna(0, inplace=True)
alldata["BsmtUnfSF"].fillna(0, inplace=True)
alldata["TotalBsmtSF"].fillna(0, inplace=True)
alldata["GarageArea"].fillna(0, inplace=True)
alldata["BsmtFullBath"].fillna(0, inplace=True)
alldata["BsmtHalfBath"].fillna(0, inplace=True)
alldata["GarageCars"].fillna(0, inplace=True)
alldata["GarageYrBlt"].fillna(0.0, inplace=True)
alldata["PoolArea"].fillna(0, inplace=True)

qual_dict = {np.nan: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
name = np.array(['ExterQual', 'PoolQC', 'ExterCond', 'BsmtQual', 'BsmtCond',
                 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond'])

for i in name:
    alldata[i] = alldata[i].map(qual_dict).astype(int)

alldata["BsmtExposure"] = alldata["BsmtExposure"].map(
    {np.nan: 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}).astype(int)

bsmt_fin_dict = {np.nan: 0, "Unf": 1, "LwQ": 2,
                 "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
alldata["BsmtFinType1"] = alldata["BsmtFinType1"].map(
    bsmt_fin_dict).astype(int)
alldata["BsmtFinType2"] = alldata["BsmtFinType2"].map(
    bsmt_fin_dict).astype(int)
alldata["Functional"] = alldata["Functional"].map(
    {np.nan: 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}).astype(int)

alldata["GarageFinish"] = alldata["GarageFinish"].map(
    {np.nan: 0, "Unf": 1, "RFn": 2, "Fin": 3}).astype(int)
alldata["Fence"] = alldata["Fence"].map(
    {np.nan: 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}).astype(int)

# encoding data
alldata["CentralAir"] = (alldata["CentralAir"] == "Y") * 1.0
varst = np.array(['MSSubClass', 'LotConfig', 'Neighborhood', 'Condition1',
                  'BldgType', 'HouseStyle', 'RoofStyle', 'Foundation', 'SaleCondition'])

for x in varst:
    factorize(alldata, x)

# encode variables and impute missing values
alldata = factorize(alldata, "MSZoning", "RL")
alldata = factorize(alldata, "Exterior1st", "Other")
alldata = factorize(alldata, "Exterior2nd", "Other")
alldata = factorize(alldata, "MasVnrType", "None")
alldata = factorize(alldata, "SaleType", "Oth")

# creating new variable (1 or 0) based on irregular count levels
# The level with highest count is kept as 1 and rest as 0
alldata["IsRegularLotShape"] = (alldata["LotShape"] == "Reg") * 1
alldata["IsLandLevel"] = (alldata["LandContour"] == "Lvl") * 1
alldata["IsLandSlopeGentle"] = (alldata["LandSlope"] == "Gtl") * 1
alldata["IsElectricalSBrkr"] = (alldata["Electrical"] == "SBrkr") * 1
alldata["IsGarageDetached"] = (alldata["GarageType"] == "Detchd") * 1
alldata["IsPavedDrive"] = (alldata["PavedDrive"] == "Y") * 1
alldata["HasShed"] = (alldata["MiscFeature"] == "Shed") * 1
alldata["Remodeled"] = (alldata["YearRemodAdd"] != alldata["YearBuilt"]) * 1

# Did the modeling happen during the sale year?
alldata["RecentRemodel"] = (alldata["YearRemodAdd"] == alldata["YrSold"]) * 1

# Was this house sold in the year it was built?
alldata["VeryNewHouse"] = (alldata["YearBuilt"] == alldata["YrSold"]) * 1
alldata["Has2ndFloor"] = (alldata["2ndFlrSF"] == 0) * 1
alldata["HasMasVnr"] = (alldata["MasVnrArea"] == 0) * 1
alldata["HasWoodDeck"] = (alldata["WoodDeckSF"] == 0) * 1
alldata["HasOpenPorch"] = (alldata["OpenPorchSF"] == 0) * 1
alldata["HasEnclosedPorch"] = (alldata["EnclosedPorch"] == 0) * 1
alldata["Has3SsnPorch"] = (alldata["3SsnPorch"] == 0) * 1
alldata["HasScreenPorch"] = (alldata["ScreenPorch"] == 0) * 1

# setting levels with high count as 1 and the rest as 0
# you can check for them using the value_counts function
alldata["HighSeason"] = alldata["MoSold"].replace({1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})
alldata["NewerDwelling"] = alldata["MSSubClass"].replace({20: 1, 30: 0, 40: 0, 45: 0, 50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0, 90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0})

alldata.shape
(2915, 100)
