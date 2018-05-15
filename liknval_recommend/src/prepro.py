#! /usr/bin/python

import numpy as np
import pandas as pd


events = pd.read_csv("../input/all/events.tsv", sep="\t")
log = pd.read_csv("../input/all/log.tsv", sep="\t")
users = pd.read_csv("../input/all/users.tsv", sep="\t")
merged_data = pd.merge(pd.merge(log, events), users)
print(merged_data.isnull().sum())
fill_female_age_upper = merged_data[merged_data["female_age_upper"].isnull()].groupby("event_id")["age"].max().reset_index()
fill_male_age_upper = merged_data[merged_data["male_age_upper"].isnull()].groupby("event_id")["age"].max().reset_index()
female_price = merged_data["female_price"].median()
male_price = merged_data["male_price"].median()
events.female_price.fillna(female_price, inplace=True)
events.male_price.fillna(male_price, inplace=True)
events.female_age_upper.fillna(35, inplace=True)
events.male_age_upper.fillna(35, inplace=True)

print(events.isnull().sum())

events.to_csv("../input/all/filled_events.csv")
