import logging
from helpers import get_final_model_results, return_X_y, remove_from_list, get_features_list, \
    get_max_no_features_count, get_result, intial_check, get_current_log_file_name, file_logger


class ForwardFeatureSelection:
    def __init__(self, classifier, dataframe, target_name, metric_obj=None, scale_obj=None, test_size=0.3,
                 min_no_features=0, max_no_features=None, log=False, variation='soft', verbose=1,
                 random_state=42, selected=[]):

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

        self.selected = selected
        logging.basicConfig(format='%(asctime)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')
        self.current_log_file_name = get_current_log_file_name()

    def core_algoritm(self, X, y, feature_list, sel, max_no_features, sco):
        feature_list_len = len(feature_list)
        selected = sel
        score = sco
        temp_selected = ''
        stop = False
        min_no_features = self.min_no_features

        for iteration in range(feature_list_len):
            logging.warning("##### {} out of {} #####".format(
                iteration+1, feature_list_len))

            features = selected + [feature_list[iteration]]

            metric_score = get_result(X, y, features, self.scale_obj, self.classifier,
                                      self.test_size, self.random_state, self.metric_obj,
                                      self.verbose)

            # print(features, metric_score)
            # print('\n')

            if self.variation == 'soft':
                if metric_score > score:
                    score = metric_score
                    selected.append(feature_list[iteration])

                    if self.log:
                        content = "{} - {} \n".format(selected, score)
                        file_logger(self.current_log_file_name, content)

                # print(max_no_features, min_no_features, len(selected))
                if len(selected) >= max_no_features:
                    break

            elif self.variation == 'hard':
                if metric_score >= score:
                    score = metric_score
                    temp_selected = feature_list[iteration]

            elif self.variation == 'hard+':
                if metric_score > score:
                    score = metric_score
                    temp_selected = feature_list[iteration]

        if self.variation == 'hard' or self.variation == 'hard+':
            if temp_selected:
                selected.append(temp_selected)

                if self.log:
                    content = "{} - {} \n".format(selected, score)
                    file_logger(self.current_log_file_name, content)
            else:
                stop = True

            if len(selected) >= max_no_features:
                stop = True

        return selected, score, stop

    def soft_forward_feature_selection(self):
        X, y = return_X_y(self.dataframe, self.target_name)
        feature_list = get_features_list(self.dataframe, self.target_name)
        feature_list = remove_from_list(feature_list, self.selected)
        max_no_features = get_max_no_features_count(
            self.dataframe, self.target_name, self.max_no_features)
        score = 0

        self.selected, score, stop = self.core_algoritm(
            X, y, feature_list, self.selected, max_no_features, score)

    def hard_forward_feature_selection(self):
        X, y = return_X_y(self.dataframe, self.target_name)
        feature_list = get_features_list(self.dataframe, self.target_name)
        feature_list = remove_from_list(feature_list, self.selected)
        feature_list_len = len(feature_list)
        max_no_features = get_max_no_features_count(
            self.dataframe, self.target_name, self.max_no_features)
        score = 0
        cnt = 0

        while len(feature_list):
            logging.warning("{} out of {}".format(
                cnt+1, feature_list_len))

            temp_selected, score, stop = self.core_algoritm(
                X, y, feature_list, self.selected, max_no_features, score)

            # print(temp_selected, score, stop)
            # print(temp_selected, score)
            # print(feature_list)

            if stop:
                break

            self.selected = temp_selected
            feature_list = remove_from_list(feature_list, self.selected)

            cnt += 1

            # print(self.selected, score, feature_list)

            print('\n')

    def run(self):
        intial_check(self.min_no_features, self.max_no_features,
                     self.scale_obj, self.dataframe, self.target_name, self.selected)

        logging.warning('STARTING {} FORWARD FEATURE SELECTION'.format(
            self.variation.upper()))

        if self.variation == 'soft':
            self.soft_forward_feature_selection()
        elif self.variation == 'hard' or self.variation == 'hard+':
            self.hard_forward_feature_selection()
        else:
            logging.error('INVALID VARIATION PASSED')

        if self.verbose > 1:
            get_final_model_results(self.dataframe, self.target_name, self.selected, self.scale_obj,
                                    self.classifier, self.test_size, self.random_state,
                                    self.metric_obj, self.verbose)

        return (self.selected)
