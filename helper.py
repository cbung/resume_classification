import pandas as pd
import datetime as dt
import difflib
import re
from sklearn.cluster import KMeans


def skills_seperator(dataframe):
    def last_null_cleaner(argm):
        dummy_list = []
        for x in argm:
            if x.__contains__(": null"):
                x = x[0:-7]
            dummy_list.append(x)
        return dummy_list

    dataframe = dataframe[~(dataframe["skills"].isnull())].reset_index().drop(columns="index")
    dataframe["skills"] = dataframe["skills"].apply(lambda x: re.sub(r"\s:\s(\d+)", " : null", x))
    dataframe["skills"] = dataframe["skills"].apply(lambda x: str(x).split(" : null,"))
    dataframe["skills"] = dataframe["skills"].apply(lambda x: last_null_cleaner(x))
    return dataframe


def skill_formating_list_operations(dataframe):
    def find_programming_language(argm):
        if any(map(argm.__contains__, [" c ", " c++ ", " c# ", " r ", " python ", " julia ", " matlab ", " macro "])):
            if argm.__contains__(" c "):
                return " c_ "
            elif argm.__contains__(" c++ "):
                return " c++_  "
            elif argm.__contains__(" c# "):
                return " c#_ "
            elif argm.__contains__(" r "):
                return " r_ "
            elif argm.__contains__(" python "):
                return " python_ "
            elif argm.__contains__(" julia "):
                return " julia_ "
            elif argm.__contains__(" matlab "):
                return " matlab_ "
            elif argm.__contains__(" macro "):
                return " macro_ "
        else:
            return argm

    def is_english(s):
        if s.isascii():
            return s
        else:
            return "XXXXXX"

    nestedSkills_ = dataframe["skills"].to_list()
    rawUniqueSkillsList_ = list(set(sum(nestedSkills_, [])))
    onlyEnglish_list_ = [is_english(x) for x in rawUniqueSkillsList_]
    onlyEnglish_list_with_dup_ = [x.lower() for x in onlyEnglish_list_ if x != "XXXXXX"]
    onlyEnglish_list_no_dup_ = list(set(onlyEnglish_list_with_dup_))
    remove_parantList_ = [re.sub("[\(\[].*?[\)\]]", "", x) for x in onlyEnglish_list_no_dup_]
    remove_colonList_ = [re.sub(r':[^.]+', '', x) for x in remove_parantList_]
    remove_hyphensList_ = [re.sub('(?<=[a-z])-(?=[a-z])', ' ', x) for x in remove_colonList_]
    remove_slashList_ = [re.sub('(?<=[a-z])/(?=[a-z])', ' ', x) for x in remove_hyphensList_]
    remove_blanksList_ = [" ".join(x.split()) for x in remove_slashList_]
    remove_blanksList_ = [x.strip() for x in remove_blanksList_]
    centered_list_ = [x.center(len(x) + 2) for x in remove_blanksList_]
    programmingLanguagesFix_ = [find_programming_language(x) for x in centered_list_]
    remove_blanksList_last_ = [" ".join(x.split()) for x in programmingLanguagesFix_]
    remove_blanksList_last_ = [x.strip() for x in remove_blanksList_last_]
    list_for_dataframe_ = list(set(remove_blanksList_last_))
    return list_for_dataframe_


def get_unique_skills(list_formated, cut_off_val=0.90, max_similars=10):
    nestedSkills_up1_ = [difflib.get_close_matches(x, list_formated, n=max_similars, cutoff=cut_off_val)
                         for x in list_formated]
    strSimilarSkills_ = [tuple(sorted(set(x))) for x in nestedSkills_up1_ if len(x) > 1]
    strSimilarSkills_ = [x for x in strSimilarSkills_ if len(x) > 1]
    strSimilarSkills_uniqueTuples_ = tuple(set(sorted(strSimilarSkills_)))
    strSimilarSkills_uniqueLists_ = list(set([tuple(x) for x in strSimilarSkills_uniqueTuples_]))

    test_list_first_ = []
    test_list_nofirst_ = []
    for x in strSimilarSkills_uniqueLists_:
        append_it_ = []
        for inner_i_, y in enumerate(x):
            if inner_i_ == 0:
                test_list_first_.append(y)
            else:
                append_it_.append(y)
                if len(append_it_) >= len(x) - 1:
                    test_list_nofirst_.append(append_it_)
    similars_dict_ = dict()
    for x, y in zip(test_list_first_, test_list_nofirst_):
        for inner_y_ in y:
            similars_dict_.update({inner_y_: x})
    dict_operationsList_ = [similars_dict_[x] if x in list(similars_dict_.keys())
                            else x for x in list_formated]

    list_for_dataframe_ = list(set(dict_operationsList_))

    return similars_dict_, list_for_dataframe_


def skill_formating_dataframe_operations(dataframe, dictionary):
    def find_programming_language(argm):
        if any(map(argm.__contains__, [" c ", " c++ ", " c# ", " r ", " python ", " julia ", " matlab ", " macro "])):
            if argm.__contains__(" c "):
                return " c_ "
            elif argm.__contains__(" c++ "):
                return " c++_  "
            elif argm.__contains__(" c# "):
                return " c#_ "
            elif argm.__contains__(" r "):
                return " r_ "
            elif argm.__contains__(" python "):
                return " python_ "
            elif argm.__contains__(" julia "):
                return " julia_ "
            elif argm.__contains__(" matlab "):
                return " matlab_ "
            elif argm.__contains__(" macro "):
                return " macro_ "
        else:
            return argm

    def is_english(s):
        if s.isascii():
            return s
        else:
            return "XXXXXX"

    dataframe["skills"] = dataframe["skills"].apply(lambda x: [is_english(in_x) for in_x in x])
    dataframe["skills"] = dataframe["skills"].apply(lambda x: [in_x.lower() for in_x in x])
    dataframe["skills"] = dataframe["skills"].apply(lambda x: [re.sub("[\(\[].*?[\)\]]", "", in_x) for in_x in x])
    dataframe["skills"] = dataframe["skills"].apply(lambda x: [re.sub(r':[^.]+', '', in_x) for in_x in x])
    dataframe["skills"] = dataframe["skills"].apply(lambda x: [re.sub('(?<=[a-z])-(?=[a-z])', ' ', in_x) for in_x in x])
    dataframe["skills"] = dataframe["skills"].apply(lambda x: [re.sub('(?<=[a-z])/(?=[a-z])', ' ', in_x) for in_x in x])
    dataframe["skills"] = dataframe["skills"].apply(lambda x: [" ".join(in_x.split()) for in_x in x])
    dataframe["skills"] = dataframe["skills"].apply(lambda x: [in_x.strip() for in_x in x])
    dataframe["skills"] = dataframe["skills"].apply(lambda x: [in_x.center(len(in_x) + 2) for in_x in x])
    dataframe["skills"] = dataframe["skills"].apply(lambda x: [find_programming_language(in_x) for in_x in x])
    dataframe["skills"] = dataframe["skills"].apply(lambda x: [" ".join(in_x.split()) for in_x in x])
    dataframe["skills"] = dataframe["skills"].apply(lambda x: [in_x.strip() for in_x in x])
    dataframe["skills"] = dataframe["skills"].apply(lambda x: [dictionary[in_x] if in_x in list(dictionary.keys()) else in_x for in_x in x])
    dataframe["skills"] = dataframe["skills"].apply(lambda x: list(set(x)))
    return dataframe


def operations_with_value_counts_lists(dataframe, list_formated, check_str="sql", summary=True, verbose=False, return_it=False, quan_perc_th=4):
    contains_list = [x for x in list_formated if x.__contains__(check_str)]
    unnecessary_contains = []
    necessary_contains = []
    for variance in contains_list:
        if (dataframe["skills"].apply(lambda ind_list: 1 if variance in ind_list else 0).value_counts()[1] / dataframe.shape[0]) * 100 > quan_perc_th:
            necessary_contains.append([variance,
                                       dataframe["skills"].apply(lambda ind_list: 1 if variance in ind_list else 0).value_counts()[1]])
        else:
            unnecessary_contains.append(variance)

    if summary:
        if len(necessary_contains) < 1:
            return [check_str, len(necessary_contains), len(unnecessary_contains)]
        else:
            return [check_str, len(necessary_contains), len(unnecessary_contains), sorted(necessary_contains, key=lambda nested: nested[1], reverse=True)[0][1]]

    if verbose:
        if len(necessary_contains) < 1:
            print(f"Summary: {check_str}\n", "--------------------------------------\n",
                  f"Full List       : {contains_list}\n",
                  f"Necessary       : Empty\n",
                  f"Unnecessary     : {unnecessary_contains}\n",
                  f"Changes into    : No Change\n",
                  f"Unnecessary List: {len(unnecessary_contains)} skills\n", )
        else:
            print(f"Summary: {check_str}\n", "--------------------------------------\n",
                  f"Full List       : {contains_list}\n",
                  f"Necessary       : {necessary_contains}\n",
                  f"Unnecessary     : {unnecessary_contains}\n",
                  f"Changes into    : {sorted(necessary_contains, key=lambda nested: nested[1], reverse=True)[0][0]},  {sorted(necessary_contains, key=lambda nested: nested[1], reverse=True)[0][1]}\n",
                  f"How many Changes: {len(unnecessary_contains)}\n", )

    if return_it:
        return unnecessary_contains, necessary_contains


def change_it_with_necessary(dataframe, list_formated, check_str="sql", quan_perc_th=4):
    unnecessary_contains_, necessary_contains_ = operations_with_value_counts_lists(dataframe=dataframe, list_formated=list_formated,
                                                                                    check_str=check_str, summary=False, verbose=False,
                                                                                    return_it=True, quan_perc_th=quan_perc_th)
    if len(necessary_contains_) > 0:
        mode_of_necessary_ = sorted(necessary_contains_, key=lambda nested: nested[1], reverse=True)[0][0]
        dict_for_change = dict()
        for x, y in zip([mode_of_necessary_], [unnecessary_contains_]):
            for inner_y in y:
                dict_for_change.update({inner_y: x})
        dict_ops_afterChange = [dict_for_change[x] if x in list(dict_for_change.keys())
                                else x for x in list_formated]
        dict_ops_afterChange = list(set(dict_ops_afterChange))

        dataframe["skills"] = dataframe["skills"].apply(lambda x_: [dict_for_change[inner_x]
                                                                    if inner_x in list(dict_for_change.keys())
                                                                    else inner_x for inner_x in x_])
        dataframe["skills"] = dataframe["skills"].apply(lambda x_: list(set(x_)))

    return dict_ops_afterChange


def create_list_of_words_for_change(list_formated, frequency=0.25):
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import collections

    # nltk.download('punkt')

    string_skills = ""
    for x in list_formated:
        string_skills = string_skills + x + " "

    text_tokens = word_tokenize(string_skills, language="english")
    tokens_without_sw = [word for word in text_tokens if word not in stopwords.words('english')]
    counter = collections.Counter(tokens_without_sw)
    counter_list = counter.most_common()
    word_list_for_change_raw = [word for word, quantity in counter_list if (quantity / len(counter_list)) * 100 >= frequency]
    word_list_for_change = [word for word in word_list_for_change_raw if (len(word) >= 3) and (word != "data")]

    return word_list_for_change


def get_list_of_word_for_change(dataframe, list_formated, quan_perc_th=4, frequency=0.25, verbose=False, change_it=False, get_list_for_diff=False):
    list_of_words = create_list_of_words_for_change(list_formated=list_formated, frequency=frequency)

    filtered_for_change_raw = []
    filtered_for_diff_ops = []
    for x in list_of_words:
        check_list = operations_with_value_counts_lists(dataframe=dataframe, list_formated=list_formated,
                                                        check_str=x, summary=True, verbose=False,
                                                        return_it=False, quan_perc_th=quan_perc_th)
        if check_list[1] != 0:
            filtered_for_change_raw.append(check_list)
        else:
            filtered_for_diff_ops.append(check_list[0])

    if get_list_for_diff:
        return filtered_for_diff_ops

    sorted_for_change_summary = sorted(filtered_for_change_raw,
                                       key=lambda nested: (nested[3], nested[2], nested[1]), reverse=False)
    sorted_words_for_change = [word[0] for word in sorted_for_change_summary]

    if verbose:
        for x in sorted_words_for_change:
            operations_with_value_counts_lists(dataframe=dataframe, list_formated=list_formated,
                                               check_str=x, summary=False, verbose=True,
                                               return_it=False, quan_perc_th=quan_perc_th)

    if change_it:
        for i, x in enumerate(sorted_words_for_change):
            if i == 0:
                changed_list_ = change_it_with_necessary(dataframe=dataframe, list_formated=list_formated,
                                                         check_str=x, quan_perc_th=quan_perc_th)
            else:
                changed_list_ = change_it_with_necessary(dataframe=dataframe, list_formated=changed_list_,
                                                         check_str=x, quan_perc_th=quan_perc_th)
        return changed_list_


def stats_for_skills(dataframe, list_formated, summary_get_all=False, summary_get_ind=False, return_all=False):
    perc_20 = sorted([[skill_, dataframe["skills"].apply(lambda x: 1 if skill_ in x else 0).value_counts()[1]]
                      for skill_ in list_formated if dataframe["skills"].apply(lambda x: 1 if skill_ in x else 0).value_counts()[1] >= round(dataframe.shape[0] * 0.20)],
                     key=lambda nested: nested[1], reverse=True)
    perc_10 = sorted([[skill_, dataframe["skills"].apply(lambda x: 1 if skill_ in x else 0).value_counts()[1]]
                      for skill_ in list_formated if
                      round(dataframe.shape[0] * 0.20) > dataframe["skills"].apply(lambda x: 1 if skill_ in x else 0).value_counts()[1] >= round(dataframe.shape[0] * 0.10)],
                     key=lambda nested: nested[1], reverse=True)
    perc_5 = sorted([[skill_, dataframe["skills"].apply(lambda x: 1 if skill_ in x else 0).value_counts()[1]]
                     for skill_ in list_formated if
                     round(dataframe.shape[0] * 0.10) > dataframe["skills"].apply(lambda x: 1 if skill_ in x else 0).value_counts()[1] >= round(dataframe.shape[0] * 0.05)],
                    key=lambda nested: nested[1], reverse=True)
    perc_1 = sorted([[skill_, dataframe["skills"].apply(lambda x: 1 if skill_ in x else 0).value_counts()[1]]
                     for skill_ in list_formated if
                     round(dataframe.shape[0] * 0.05) > dataframe["skills"].apply(lambda x: 1 if skill_ in x else 0).value_counts()[1] >= round(dataframe.shape[0] * 0.01)],
                    key=lambda nested: nested[1], reverse=True)
    perc_del = sorted([[skill_, dataframe["skills"].apply(lambda x: 1 if skill_ in x else 0).value_counts()[1]]
                       for skill_ in list_formated if round(dataframe.shape[0] * 0.01) > dataframe["skills"].apply(lambda x: 1 if skill_ in x else 0).value_counts()[1] >= 1],
                      key=lambda nested: nested[1], reverse=True)

    all_lists = [perc_20, perc_10, perc_5, perc_1, perc_del]

    if summary_get_all:
        print("----------------------------------------------------------------------")
        print(f"Number of Skills                                    : {len(sum(all_lists, []))}")
        print(f"Number of skills that more than 20% of people have  : {len(perc_20)}")
        print(f"Number of skills that between 10-20% of people have : {len(perc_10)}")
        print(f"Number of skills that between 5-10% of people have  : {len(perc_5)}")
        print(f"Number of skills that between 1-5% of people have   : {len(perc_1)}")
        print(f"Number of totally unnecessary skills                : {len(perc_del)}")
        print("----------------------------------------------------------------------")

    if summary_get_ind:
        for segment_index, perc_list in enumerate(all_lists):
            print("--------------------------------------------------------------------------------------------")
            print(f"Frequency Segment - {segment_index + 1}")
            print("--------------------------------------------------------------------------------------------")
            for inner_counter, inner_list in enumerate(perc_list):
                if inner_counter < 25:
                    print(f"Skill     : {inner_list[0]}\n"
                          f"Frequency : {inner_list[1]}\n"
                          f"Ratio     : {inner_list[1] / dataframe.shape[0] * 100:.2f}\n")
                    print("----------------------------------------------")
            print(f"There are {max(0, inner_counter - 25)} more skills in this segment")

    if return_all:
        return all_lists


def create_skill_cols_main(dataframe, list_formated, column_best=False, column_mid=False, column_all_noticable=False, return_list=False, save_dataframe=False):
    perc_lists = stats_for_skills(dataframe=dataframe, list_formated=list_formated, summary_get_ind=False, summary_get_all=False, return_all=True)

    sent_list = []
    if column_best:
        sent_list = perc_lists[0]
    if column_mid:
        sent_list = perc_lists[0] + perc_lists[1]
    if column_all_noticable:
        sent_list = perc_lists[0] + perc_lists[1] + perc_lists[2]

    for ind_skill, quantity in sent_list:
        newframe = pd.DataFrame()
        newframe[f"skill_{ind_skill}".upper()] = dataframe["skills"].apply(lambda x: 1 if ind_skill in x else 0)
        dataframe = pd.concat([dataframe, newframe], axis=1)

    if return_list:
        return [[ind_skill, quantity] for ind_skill, quantity in sent_list]
    if save_dataframe:
        return dataframe


def skill_operations(dataframe):
    dataframe = skills_seperator(dataframe=dataframe)
    similars_dict, formated_list = get_unique_skills(list_formated=skill_formating_list_operations(dataframe),
                                                     cut_off_val=0.90, max_similars=10)
    dataframe = skill_formating_dataframe_operations(dataframe=dataframe, dictionary=similars_dict)
    changed_list = get_list_of_word_for_change(dataframe=dataframe, list_formated=formated_list,
                                               quan_perc_th=2, frequency=0.25, verbose=False, change_it=True,
                                               get_list_for_diff=False)
    dataframe = create_skill_cols_main(dataframe=dataframe, list_formated=changed_list,
                                       column_best=False, column_mid=True, column_all_noticable=False,
                                       return_list=False, save_dataframe=True)
    return dataframe


def create_experience_level(dataframe):
    import warnings
    warnings.filterwarnings("ignore")

    changeToDatetime_list = [col for col in dataframe.columns if (col.__contains__("_end_"))
                             or (col.__contains__("_start_"))]

    for col in changeToDatetime_list:
        if col.__contains__("organization"):
            dataframe[col] = dataframe[col].apply(lambda x: float(str(x) + "1" if str(x).endswith("0") else x))
            dataframe[col] = dataframe[col].apply(lambda x: str(x).replace(".", "-"))
            dataframe[col] = dataframe[col].replace("nan", str([dt.date.today()][0])[:-3])
            dataframe[col] = pd.to_datetime(dataframe[col], format="%Y-%m")

        else:
            dataframe[col] = dataframe[col].apply(lambda x: float(str(x) + "1" if str(x).endswith("0") else x))
            dataframe[col] = dataframe[col].apply(lambda x: str(x).replace(".", "-"))
            dataframe[col] = dataframe[col].replace("nan", str([dt.date.today()][0])[:-3])
            dataframe[col] = pd.to_datetime(dataframe[col], format="%Y-%m")

    alldate_cols = [col for col in dataframe.columns if ("organization" in col) and (("start" in col) or ("end" in col))]
    alldate_df = dataframe[alldate_cols]

    for i in range(4):
        alldate_df[f"time_worked_{i + 1}"] = alldate_df[f"organization_end_{i + 1}"] - alldate_df[f"organization_start_{i + 1}"]

    time_worked_cols = [col for col in alldate_df.columns if "time_worked_" in col]

    for col in time_worked_cols:
        alldate_df[col] = alldate_df[col].apply(lambda x: re.findall(r'\d+', str(x)))
        alldate_df[col] = alldate_df[col].apply(lambda x: int(int(x[0]) / 30))

    alldate_df["time_worked_total"] = alldate_df["time_worked_1"] + alldate_df["time_worked_2"] + alldate_df["time_worked_3"] + alldate_df["time_worked_4"]

    alldate_df.loc[(alldate_df["time_worked_total"] >= 0) & (alldate_df["time_worked_total"] <= 24), "NEW_EXPERIENCE_LEVEL"] = "Junior"
    alldate_df.loc[(alldate_df["time_worked_total"] > 24) & (alldate_df["time_worked_total"] <= 60), "NEW_EXPERIENCE_LEVEL"] = "Mid"
    alldate_df.loc[(alldate_df["time_worked_total"] > 60) & (alldate_df["time_worked_total"] <= 120), "NEW_EXPERIENCE_LEVEL"] = "Senior"
    alldate_df.loc[(alldate_df["time_worked_total"] > 120) & (alldate_df["time_worked_total"] <= alldate_df["time_worked_total"].max()), "NEW_EXPERIENCE_LEVEL"] = "Master"

    dataframe["NEW_EXPERIENCE_LEVEL"] = alldate_df["NEW_EXPERIENCE_LEVEL"]

    return dataframe


def create_highest_degree(dataframe):
    degree_cols = [col for col in dataframe.columns if "n_degree_" in col]
    degree_df = dataframe[degree_cols]

    degree_df = degree_df.applymap(lambda x: str(x).lower())
    degree_df = degree_df.applymap(lambda x: "bachelor" if "bachelor" in x else x)
    degree_df = degree_df.applymap(lambda x: "phd" if ("ph.d" in x) or ("phd" in x) else x)
    degree_df = degree_df.applymap(lambda x: "doctor" if ("doctor" in x) or ("phd" in x) else x)
    degree_df = degree_df.applymap(lambda x: "master" if "master" in x else x)
    degree_df = degree_df.applymap(lambda x: "other" if not (("bachelor" == x) or
                                                             ("doctor" == x) or
                                                             ("master" == x)) else x)

    degree_df = degree_df.applymap(lambda x: "1" if "other" == x else x)
    degree_df = degree_df.applymap(lambda x: "1" if "bachelor" == x else x)
    degree_df = degree_df.applymap(lambda x: "2" if "doctor" == x else x)
    degree_df = degree_df.applymap(lambda x: "3" if "master" == x else x)

    degree_df["highest_degree"] = degree_df["education_degree_1"] + " " + degree_df["education_degree_2"] + " " + degree_df["education_degree_3"]
    degree_df["highest_degree"] = degree_df["highest_degree"].apply(lambda x: sorted(x.split(" "), reverse=True))
    degree_df["highest_degree"] = degree_df["highest_degree"].apply(lambda x: int(x[0]))

    degree_df["highest_degree"] = degree_df["highest_degree"].apply(lambda x: "bachelor" if x == 1 else x)
    degree_df["highest_degree"] = degree_df["highest_degree"].apply(lambda x: "doctor" if x == 2 else x)
    degree_df["highest_degree"] = degree_df["highest_degree"].apply(lambda x: "master" if x == 3 else x)
    degree_df["highest_degree"] = degree_df["highest_degree"].apply(lambda x: "other" if x == 0 else x)

    dataframe["NEW_HIGHEST_DEGREE"] = degree_df["highest_degree"]

    return dataframe


def get_cluster_col(dataframe, number_of_clusters=10, cluster_mean=False, cluster_count=False, save_it=False):
    import warnings
    warnings.filterwarnings("ignore")

    model_df_skill = dataframe[[col for col in dataframe.columns if col.__contains__("SKILL_")]]
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=17).fit(model_df_skill)
    clusters_kmeans = kmeans.labels_
    model_df_skill["kmeans_cluster_no"] = clusters_kmeans

    if cluster_mean:
        print(model_df_skill.groupby("kmeans_cluster_no").agg(["mean"]))
    if cluster_count:
        print(model_df_skill.groupby("kmeans_cluster_no").agg(["count"]))

    if save_it:
        dataframe["SEGMENT"] = model_df_skill["kmeans_cluster_no"]
        return dataframe


def data_preperation_for_model(dataframe):
    dataframe = skill_operations(dataframe)
    dataframe = create_highest_degree(dataframe=dataframe)
    dataframe = create_experience_level(dataframe=dataframe)
    dataframe = get_cluster_col(dataframe=dataframe, number_of_clusters=10, cluster_mean=False, cluster_count=False, save_it=True)
    dataframe = dataframe[[col for col in dataframe.columns if (col.__contains__("SKILL_")) or (col.__contains__("NEW_") or col == "SEGMENT")]]
    dataframe = pd.get_dummies(dataframe, columns=[col for col in dataframe.columns if (col.__contains__("NEW_"))], drop_first=True)
    X = dataframe[[col for col in dataframe.columns if (col.__contains__("SKILL_")) or (col.__contains__("NEW_"))]]
    y = dataframe["SEGMENT"]
    return X, y, dataframe
