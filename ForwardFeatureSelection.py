import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class ForwardFeatureSelection:
    def __init__(self, classifier, dataframe, target_name, metric_obj=None, scale_obj=None, test_size=0.3,
                 min_no_features=0, max_no_features=None, log=False, variation='soft', verbose=1,
                 random_state=42):

        self.classifier = classifier
        self.dataframe = dataframe
        self.target_name = target_name
        self.scale_obj = scale_obj
        self.metric_obj = metric_obj
        self.test_size = test_size
        self.min_no_features = min_no_features
        self.max_no_features = max_no_features
        self.log = log
        self.variation = variation
        self.verbose = verbose
        self.random_state = random_state

        self.selected = []
        logging.basicConfig(format='%(asctime)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')

    def remove_from_list(self, master_list=[], remove_list=[]):
        # This function is used to remove a list of
        # values from another list

        for items in remove_list:
            if items in master_list:
                master_list.remove(items)

        return master_list

    def get_max_no_features_count(self):
        # This function is used to get the column count
        # if the max_no_features = None

        feature_list = self.get_features_list()
        if self.max_no_features == None:
            column_count = self.dataframe[feature_list].shape[1]
            return column_count

        return self.max_no_features

    def get_features_list(self):
        # This funtion return the feature list of
        # the dataframe by removing the target feature

        feature_list = list(self.dataframe)
        feature_list = self.remove_from_list(
            feature_list, [self.target_name])

        return feature_list

    def return_X_y(self):
        # This function return,
        # X - In-dependent variable dataframe
        # y - Dependent variable dataframe

        feature_list = self.get_features_list()
        X = self.dataframe[feature_list]
        y = self.dataframe[self.target_name]

        return X, y

    def build_model(self, classifier, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        return y_test, y_pred

    def calcuate_prefered_metric(self, y_test, y_pred):
        if self.metric_obj == None:
            report = classification_report(y_test, y_pred, output_dict=True)
            if self.verbose > 2:
                print(classification_report(y_test, y_pred))
            return report['accuracy']
        else:
            logging.error('WORKING ON IT')

    # def get_final_features(self):
    #     # print(self.selected)
    #     return self.selected

    def core_algoritm(self, X, y, feature_list, sel, max_no_features, sco):
        feature_list_len = len(feature_list)
        selected = sel
        score = sco
        temp_selected = ''
        stop = False

        for iteration in range(feature_list_len):
            logging.warning("##### {} out of {} #####".format(
                iteration+1, feature_list_len))

            features = selected + [feature_list[iteration]]
            X_ffs = X[features]

            # print(features)

            if self.scale_obj:
                X_ffs = self.scale_obj.fit_transform(X_ffs)

            y_test, y_pred = self.build_model(self.classifier, X_ffs, y)

            metric_score = self.calcuate_prefered_metric(y_test, y_pred)

            # print(features, metric_score)
            # print('\n')

            if self.variation == 'soft':
                if metric_score > score:
                    score = metric_score
                    selected.append(feature_list[iteration])

                if len(selected) <= max_no_features and len(selected) >= max_no_features:
                    break

            elif self.variation == 'hard':
                if metric_score > score:
                    score = metric_score
                    temp_selected = feature_list[iteration]

        if self.variation == 'hard':
            if temp_selected:
                selected.append(temp_selected)
            else:
                stop = True

        return selected, score, stop

    def soft_forward_feature_selection(self):
        X, y = self.return_X_y()
        feature_list = self.get_features_list()
        max_no_features = self.get_max_no_features_count()
        score = 0

        self.selected, score, stop = self.core_algoritm(
            X, y, feature_list, self.selected, max_no_features, score)

    def hard_forward_feature_selection(self):
        X, y = self.return_X_y()
        feature_list = self.get_features_list()
        feature_list_len = len(feature_list)
        max_no_features = self.get_max_no_features_count()
        score = 0
        cnt = 0

        while len(feature_list):
            logging.warning("{} out of {}".format(
                cnt+1, feature_list_len))

            temp_selected, score, stop = self.core_algoritm(
                X, y, feature_list, self.selected, max_no_features, score)

            # print(temp_selected, score, stop)
            # print(feature_list)

            if stop:
                break

            self.selected = temp_selected
            feature_list = self.remove_from_list(feature_list, self.selected)

            cnt += 1

            # print(self.selected, score, feature_list)

            print('\n')

    def intial_check(self):
        if self.min_no_features > self.max_no_features:
            logging.error('MINIMUM NUMBER OF FEATURES PARAMETER SHOULD \
                                BE LESS THAT MAXIMUM NUMBER OF FEATURE PARAMETER')
            exit(0)
        if self.scale_obj != None and not isinstance(self.scale_obj, object):
            logging.error('INVALID SCALER OBJECT')
            exit(0)

    def get_final_model_results(self):
        X, y = self.return_X_y()

        logging.warning("FINAL MODEL RESULTS WITH FEATURES - {}".format(
            self.selected))

        X_ffs = X[self.selected]

        if self.scale_obj:
            X_ffs = self.scale_obj.fit_transform(X_ffs)

        y_test, y_pred = self.build_model(self.classifier, X_ffs, y)

        metric_score = self.calcuate_prefered_metric(y_test, y_pred)

        logging.warning("RESULT : {}".format(metric_score))

    def run(self):
        self.intial_check()

        logging.warning('STARTING {} FORWARD FEATURE SELECTION'.format(
            self.variation.upper()))

        if self.variation == 'soft':
            self.soft_forward_feature_selection()
        elif self.variation == 'hard':
            self.hard_forward_feature_selection()
        else:
            logging.error('INVALID VARIATION PASSED')

        if self.verbose > 1:
            self.get_final_model_results()

        return (self.selected)
