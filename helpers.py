import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def intial_check(min_no_features, max_no_features, scale_obj, dataframe, target_name, selected):
    feature_list = get_features_list(dataframe, target_name)

    if min_no_features > max_no_features:
        logging.error('MINIMUM NUMBER OF FEATURES PARAMETER SHOULD \
                                BE LESS THAT MAXIMUM NUMBER OF FEATURE PARAMETER')
        exit(0)

    if scale_obj != None and not isinstance(scale_obj, object):
        logging.error('INVALID SCALER OBJECT')
        exit(0)

    for feat in selected:
        if feat not in feature_list:
            logging.error("FEATURE '{}' MISSING IN DATAFRAME".format(feat))
            exit(0)


def remove_from_list(master_list=[], remove_list=[]):
    # This function is used to remove a list of
    # values from another list

    for items in remove_list:
        if items in master_list:
            master_list.remove(items)

    return master_list


def get_features_list(dataframe, target_name):
    # This funtion return the feature list of
    # the dataframe by removing the target feature

    feature_list = list(dataframe)
    feature_list = remove_from_list(feature_list, [target_name])

    return feature_list


def get_max_no_features_count(dataframe, target_name, max_no_features):
    # This function is used to get the column count
    # if the max_no_features = None

    feature_list = get_features_list(dataframe, target_name)
    if max_no_features == None:
        column_count = dataframe[feature_list].shape[1]
        return column_count

    return max_no_features


def return_X_y(dataframe, target_name):
    # This function return,
    # X - In-dependent variable dataframe
    # y - Dependent variable dataframe

    feature_list = get_features_list(dataframe, target_name)
    X = dataframe[feature_list]
    y = dataframe[target_name]

    return X, y


def build_model(classifier, X, y, test_size, random_state):
    # This function is used to build the ML model

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    return y_test, y_pred


def calcuate_prefered_metric(y_test, y_pred, metric_obj, verbose, features):
    # This function is used to calculate the
    # prefered metric. It also has the ability to get
    # callback function

    if metric_obj == None:
        report = classification_report(y_test, y_pred, output_dict=True)
        if verbose > 2:
            print("FEATURES : {}".format(features))
            print(classification_report(y_test, y_pred))
        return report['accuracy']
    else:
        logging.error('WORKING ON IT')


def get_result(X, y, features, scale_obj, classifier, test_size, random_state,
               metric_obj, verbose):
    # Return the prefered Metric score

    X_ffs = X[features]

    if scale_obj:
        X_ffs = scale_obj.fit_transform(X_ffs)

    y_test, y_pred = build_model(classifier, X_ffs, y, test_size, random_state)

    metric_score = calcuate_prefered_metric(
        y_test, y_pred, metric_obj, verbose, features)

    return metric_score


def get_final_model_results(dataframe, target_name, selected, scale_obj, classifier,
                            test_size, random_state, metric_obj, verbose):

    X, y = return_X_y(dataframe, target_name)

    logging.warning("FINAL MODEL RESULTS WITH FEATURES - {}".format(selected))

    metric_score = get_result(X, y, selected, scale_obj, classifier, test_size,
                              random_state, metric_obj, verbose)

    logging.warning("RESULT : {}".format(metric_score))
