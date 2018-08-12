

#pipeline is used to chain multiple estimators into one
#You only have to call fit and predict once on your data to fit a whole sequence of estimators.
#The Pipeline is built using a list of (key, value) pairs, where the key is a string containing the name you want to give this step and value is an estimator object:

#An estimator is any object that learns from data; it may be a classification, regression or clustering algorithm or a transformer that extracts/filters useful features from raw data.
#
# All estimator objects expose a fit method that takes a dataset (usually a 2-d array):
# Good Read http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/

# Steps to create a estimator :
    # 1. e. You need to choose one of these: Classifier, Clusterring, Regressor and Transformer.
    # 2. After you decided which one suits to your needs you subclass BaseEstimator and an appropriate class for your type (one of ClassifierMixin, RegressorMixin, ClusterMixin, TransformerMixin).
    # 3. Now you need of course to decide what parameters an estimator receives. Decide what input it takes and what output it returns. It's all you need for now.



    #   3.1 It's good to know some additional rules.
    #      All arguments of __init__must have default value, so it's possible to initialize the classifier just by typing MyClassifier()
    #      No confirmation of input parameters should be in __init__ method! That belongs to fit method.
    #      Do not take data as argument here! It should be in fit method.


    #get_params and set_params
    #All estimators must have get_params and set_params functions. They are inherited when you subclass BaseEstimator and I would recommend not to override these function (just not state them in definition of your classifier).

    # fit method
    # In fit method you should implement all the hard work. At first you should check the parameters. Secondly, you should take and process the data.
    # You'll almost surely want add some new attributes to your object which are created in fit method. These should be ended by _ at the end, e.g. self.fitted_.
    # And finally you should return self. This is again for compatibility reasons with common interface of scikit-learn.



    # Response y vector
    #There might be cases when you do not need to input a response vector (as in example below). Nevertheless, for implementation reasons you need to add this vector to your definitions in case you want to use GridSearch and so. It's good to initialize it with None value.

from sklearn.base import BaseEstimator, ClassifierMixin
class MeanClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, intValue=0,stringParam="defaultValue", otherParam=None):
        """
       This is called when MeanClassifier is instantiate
        """
        self.intValue=intValue
        self.stringParam= stringParam

    def fit(self,x,y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """
        assert (type(self.intValue) == int), "intValue parameter must be integer"
        assert (type(self.stringParam) == str), "stringValue parameter must be string"
        self.treshold_ = (sum(x)/len(x)) + self.intValue  # mean + intValue
        return self

    def _meaning(self, x):
        # returns True/False according to fitted classifier
        # notice underscore on the beginning
        return (True if x >= self.treshold_ else False)

    def predict(self, X, y=None):
        try:
            getattr(self, "treshold_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return([self._meaning(x) for x in X])

    def score(self, X, y=None):
        # counts number of values bigger than mean
        return(sum(self.predict(X)))


from sklearn.grid_search import GridSearchCV

X_train = [i for i in range(0, 100, 5)]
X_test = [i + 3 for i in range(-5, 95, 5)]
tuned_params = {"intValue" : [-10,-1,0,1,10]}

gs = GridSearchCV(MeanClassifier(), tuned_params)

# for some reason I have to pass y with same shape
# otherwise gridsearch throws an error. Not sure why.
gs.fit(X_test, y=[1 for i in range(20)])

gs.best_params_ # {'intValue': -10} # and that is what we expect :)