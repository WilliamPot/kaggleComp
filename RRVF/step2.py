# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:07:37 2018

@author: Chen
"""

import numpy as np
import pandas as pds
import collections

def labelData(data, column):
    re = collections.Counter(data[column].values)
    re = [key for key in re]
    ran = normalization(list(range(1,len(re)+1)))
    re_dict = {}
    for i in range(len(ran)):
        re_dict[re[i]] = ran[i]
    return re_dict    
def normalization(data):
    nmin = min(data)
    nmax = max(data)
    rd = [(n-nmin)/(nmax-nmin) for n in data]
    return rd
data = pds.read_csv('processed.csv')
air_info = pds.read_csv('air_store_info.csv')
date_info = pds.read_csv('date_info.csv')
air_info.rename(columns={'air_store_id': 'store_id', 'air_genre_name': 'genre_name', \
'air_area_name': 'area_name'}, inplace=True)
hpg_info = pds.read_csv('hpg_store_info.csv')
hpg_info.rename(columns={'hpg_store_id': 'store_id', 'hpg_genre_name': 'genre_name', \
'hpg_area_name': 'area_name'}, inplace=True)
air_merged= pds.merge(data, air_info)
hpg_merged= pds.merge(data, hpg_info)
merged_data = pds.concat([air_merged,hpg_merged],ignore_index = True)
#merged_data= pds.merge(merged_data, date_info,left_on = 'datetime',right_on='')
months = np.array([int(datetime[5:7]) for datetime in merged_data['datetime'].values])
days = [int(datetime[8:10]) for datetime in merged_data['datetime'].values]
merged_data['month'] = normalization(months)
merged_data['day'] = days
           
genre_name_dict = labelData(pds.concat([air_info,hpg_info],ignore_index = True),'genre_name')
area_name_dict = labelData(pds.concat([air_info,hpg_info],ignore_index = True),'area_name')

merged_data = pds.merge(merged_data, date_info, left_on='datetime', right_on='calendar_date')
merged_data = merged_data.drop(['datetime'],axis=1)
merged_data['area_name'] = [area_name_dict[area_name] for area_name in merged_data['area_name'].values]
merged_data['genre_name'] = [genre_name_dict[genre_name] for genre_name in merged_data['genre_name'].values]
merged_data['longitude'] = normalization(merged_data['longitude'].values)
merged_data['latitude'] = normalization(merged_data['latitude'].values)
merged_data = merged_data.drop(['store_id','calendar_date'],axis=1)
day_dict = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
merged_data['day_of_week'] = normalization([day_dict[day] for day in merged_data['day_of_week'].values])
#merged_data.index  = merged_data['reserve_visitors']
#merged_data = merged_data.drop(['reserve_visitors'],axis=1)
merged_data.to_csv('train_data.csv')

test_data = pds.read_csv('sample_submission.csv')
test_prefix = [air_id for air_id in test_data['id'].values]
test_air_id = [air_id[0:20] for air_id in test_data['id'].values]
test_date = [air_id[21:] for air_id in test_data['id'].values]
test_months = [int(air_id[26:28]) for air_id in test_data['id'].values]
test_days = [int(air_id[29:]) for air_id in test_data['id'].values]
test_data['store_id'] = test_air_id
test_data['datetime'] = test_date
test_data['month'] = [0.273 if(month==4) else 0.364 for month in test_months]
test_data['day'] = test_days
test_data = pds.merge(test_data, air_info)
test_data = pds.merge(test_data, date_info, left_on='datetime', right_on='calendar_date')
test_data = test_data.drop(['datetime','calendar_date','id','visitors'],axis=1)
test_data['area_name'] = [area_name_dict[area_name] for area_name in test_data['area_name'].values]
test_data['genre_name'] = [genre_name_dict[genre_name] for genre_name in test_data['genre_name'].values]
test_data['longitude'] = normalization(test_data['longitude'].values)
test_data['latitude'] = normalization(test_data['latitude'].values)
test_data['store_id'] = test_prefix
test_data['day_of_week'] = normalization([day_dict[day] for day in test_data['day_of_week'].values])
#test_data.index  = test_data['area_name']
#test_data = test_data.drop(['area_name'],axis=1)
test_data.to_csv('test_data.csv')