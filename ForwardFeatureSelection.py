
class ForwardFeatureSelection:
    def __init__(self, classifier, X_train, y_train, X_test, y_test, min_no_features=0, max_no_features=None, log=False, verbose=1):
        self.classifier = classifier
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.min_no_features = min_no_features
        self.max_no_features = max_no_features
        self.log = log
        self.verbose = verbose

        def get_max_no_features(self):
            if self.max_no_features == None:
                column_count = self.X_train.shape[1]
                return column_count
            return self.max_no_features
