def user_join_list(log):
    return log[log["action_type"] == 3]


def join_event_filter(user_id, event_list, join_list):
  user_join_list = join_list[join_list["user_id"] == user_id].event_id
  print(user_join_list)
  not_join_event_list = event_list[~event_list["event_id"].isin(user_join_list)]
  return not_join_event_list

def age_filter(user_id, join_event_list, event_details):
    return join_event_list
