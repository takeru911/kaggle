#! /usr/bin/python

import numpy as np
import pandas as pd

def fill_na(merged_data):
# eventのnaデータを補完していく
# female_age_uppser, male_age_usser, female_price, male_priceあたり
  fill_female_age_upper = merged_data[merged_data["female_age_upper"].isnull()].groupby("event_id")["age"].max().reset_index()
  fill_male_age_upper = merged_data[merged_data["male_age_upper"].isnull()].groupby("event_id")["age"].max().reset_index()
  female_price = merged_data["female_price"].median()
  male_price = merged_data["male_price"].median()
  merged_data.female_price.fillna(female_price, inplace=True)
  merged_data.male_price.fillna(male_price, inplace=True)
  merged_data.female_age_upper.fillna(35, inplace=True)
  merged_data.male_age_upper.fillna(35, inplace=True)

#  for index,d in merged_data.iterrows():
#      if index % 1000 == 0:
#          print("now iter: {}".format(index))
#      if d.event_id in list(fill_female_age_upper.event_id):
#          d.femail_age_upper = fill_female_age_upper[fill_female_age_upper["event_id"] == d.event_id].age
#      if d.event_id in list(fill_male_age_upper.event_id):
#          d.mail_age_upper = fill_male_age_upper[fill_male_age_upper["event_id"] == d.event_id].age

  return merged_data


events = pd.read_csv("../input/all/events.tsv", sep="\t")
log = pd.read_csv("../input/all/log.tsv", sep="\t")
users = pd.read_csv("../input/all/users.tsv", sep="\t")
all = pd.merge(pd.merge(log, users),events)
print(all.isnull().sum())
merged_data = fill_na(all)
print(merged_data.isnull().sum())

merged_data.to_csv("../input/prepro.csv")
