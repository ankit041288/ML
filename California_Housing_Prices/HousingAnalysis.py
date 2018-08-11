import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import FeatureUnion

from California_Housing_Prices import CombinedAttributesAdder
from California_Housing_Prices import DataFrameSelector

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression




# Load CSV file in pandas
def load_housing_data(csv_path="/Users/axb4725/PycharmProjects/ML/California_Housing_Prices/Data/housing.csv"):
    return pd.read_csv(csv_path)

# Test loaded data
housing = load_housing_data()
print(housing.head())
print(housing.info())

# Quick Plot
housing.hist(bins=50, figsize=(20,15))
plt.show()



# Important step ---- Creating test set ----
#This is good but every time you run this you will get a different permutation and your model will see the whole data

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)


# better way to create test data

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

# housind data does not have index
housing_with_id = housing.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

# OR avoid all this and use Scikit learning

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)



#Suppose you chatted with experts who told you that the median income is a very important attribute to predict median housing prices. You may want to ensure that the test set is representative of the various categories of incomes in the whole dataset. Since the median income is a continuous numerical attribute, you first need to create an income category attribute. Let’s look at the median income histogram more closely (back in Figure 2-8): most median income values are clustered around $20,000–$50,000, but some median incomes go far beyond $60,000. It is important to have a sufficient number of instances in your dataset for each stratum, or else the estimate of the stratum’s importance may be biased. This means that you should not have too many strata, and each stratum should be large enough. The following code creates an income category attribute by dividing the median income by 1.5 (to limit the number of income categories), and rounding up using ceil (to have discrete categories), and then merging all the categories greater than 5 into category 5:
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


#Visualize
housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y = "latitude")

#To view Density in graph
housing.plot(kind="scatter", x="longitude", y = "latitude", alpha=0.1)


# Color Map
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()



# Looking at co-relations
corr_matrix = housing.corr()


# Part of cleaning is to remove properties .. or .. fill empty values in vectors
imputer = Imputer(strategy="median")

#Since the median can only be computed on numerical attributes, we need to create a copy of the data without the text attribute ocean_proximity:
housing_num = housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)

#Now you can use this “trained” imputer to transform the training set by replacing missing values by the learned medians:
X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns)

#Handling Test And Categorical Attributes
housing_cat = housing["ocean_proximity"]
housing_cat.head()

#Converting category to numbers
housing_cat_encode, housing_categories = housing_cat.factorize()
print(housing_cat_encode[:10])
print(housing_categories)


# SkLearn One Hot Vector
#Scikit-Learn provides a OneHotEncoder encoder to convert integer categorical values into one-hot vectors.
# One hot vector meaning only one "1" per row .. with N Columns
#One hot encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction.
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encode.reshape(-1,1))
housing_cat_1hot

#If a categorical attribute has a large number of possible categories
# (e.g., country code, profession, species, etc.), then one-hot encoding will result in a
# large number of input features. This may slow down training and degrade performance.
#  If this happens, you will want to produce denser representations called embeddings,
# but this requires a good understanding of neural networks (see Chapter 14 for more details).
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


#As you can see, there are many data transformation steps
#  that need to be executed in the right order. Fortunately,
# Scikit-Learn provides the Pipeline class to help with such sequences
# of transformations. Here is a small pipeline for the numerical attributes:



#The Pipeline constructor takes a list of name/estimator pairs defining a sequence of steps
num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
#When you call the pipeline’s fit() method, it calls fit_transform()
# sequentially on all transformers, passing the output of each call as
# the parameter to the next call, until it reaches the final estimator,
# for which it just calls the fit() method.

housing_num_tr = num_pipeline.fit_transform(housing_num)


num_attributes = list(housing_num)
cat_attributes = ["ocean_proximity"]
num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attributes)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attributes)),
        ('cat_encoder', OneHotEncoder(encoding="onehot-dense")),
    ])

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])




housing_prepared = full_pipeline.fit_transform(housing)


# Select And train Model
lin_reg = LinearRegression()
#lin_reg.fit(housing_prepared, housing_labels)



