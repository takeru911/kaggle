import pandas as pd

def user_join_list(log):
    return log[log["action_type"] == 3]


def join_event_filter(user_id, event_list, join_list):
  user_join_list = join_list[join_list["user_id"] == user_id].event_id
  not_join_event_list = event_list[~event_list["event_id"].isin(user_join_list)]
  return not_join_event_list


def age_filter(user_id, join_event_list, event_details, user_detail):
    age = user_detail[user_detail["user_id"] == user_id]["age"][user_id - 1]
    user_gender = user_detail[user_detail["user_id"] == user_id]["gender"][user_id - 1]
    join_event_detail = pd.merge(event_details, join_event_list)
    if user_gender == "male":
        tmp =join_event_detail[((join_event_detail["male_age_lower"] < age) & (join_event_detail["male_age_upper"] > age))]
        return tmp[["event_id", "rating"]]
    else:
        tmp = join_event_detail[((join_event_detail["female_age_lower"] < age) & (join_event_detail["female_age_upper"] > age))]
        return tmp[["event_id", "rating"]]


def ranking(join_event_list):
    rank = join_event_list.rank(ascending=False, method="first")
    join_event_list["ranking"] = rank["rating"]
    return join_event_list

