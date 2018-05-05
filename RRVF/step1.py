# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:03:31 2018

@author: Chen
"""

import numpy as np
import pandas as pds

def ReadAndDrop(filename):
    res = pds.read_csv(filename)
    res = res.drop(['reserve_datetime'],axis=1)
    #ids = res[filename[0:3]+'_store_id']
    #res= res.drop([filename[0:3]+'_store_id'],axis = 1)
    #res.index = ids.values
    return res
def visitTimesCum(df,prefix):
    df= df.groupby(by=[prefix+'_store_id','visit_datetime'])['reserve_visitors'].sum()
    df = df.to_frame()
    return df
relations = pds.read_csv('store_id_relation.csv')
air_res = ReadAndDrop('air_reserve.csv')
hpg_res = ReadAndDrop('hpg_reserve.csv')
hpg_info = pds.read_csv('hpg_store_info.csv')
air_res= visitTimesCum(air_res,'air')
hpg_res= visitTimesCum(hpg_res,'hpg')
air_id = [index[0] for index in air_res.index]
air_date = [index[1][0:10] for index in air_res.index]
air_res['store_id'] = air_id
air_res['datetime'] = air_date
       
       
hpg_id = [index[0] for index in hpg_res.index]
hpg_date = [index[1][0:10] for index in hpg_res.index]
hpg_res['store_id'] = hpg_id
hpg_res['datetime'] = hpg_date
       
       
new_hpg_id = [relations[relations['hpg_store_id'] == hpg_id]['air_store_id'].values[0]\
                 if(hpg_id in relations.hpg_store_id.values and hpg_id not in hpg_info.hpg_store_id.values)\
                 else hpg_id for hpg_id in hpg_res.store_id.values]
hpg_res['store_id'] = new_hpg_id
merged_res = pds.concat([air_res,hpg_res],ignore_index = True)


merged_res = merged_res.groupby(by=['store_id','datetime'])['reserve_visitors'].sum()
merged_res = merged_res.to_frame()
#merged_id = [index[0] for index in merged_res.index]
#merged_date = [index[1] for index in merged_res.index]
#merged_res['store_id'] = merged_id
#merged_res['datetime'] = merged_date
merged_res.to_csv('processed.csv')