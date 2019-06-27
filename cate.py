# 0 从product表中构造出cate的基本属性
# 1 对cate的下单记录中 不同年龄用户记录数量的比重 & 该年龄段的下单记录中，购买该cate的记录比重
# 2 对cate的下单记录中 不同性别用户记录数量的比重 & 。。。。。。
# 3 对cate的下单记录中 不同城市等级用户记录数量的比重 & 。。。。。。
# 4 对cate的下单记录中 不同会员等级用户记录数量的比重 & 。。。。。。
# 5 原来直接过来的  cate 的行为方差 日均值 转化率     【hc】
# 6 最近该cate的销量 / 总的销量【是否当时促销】         【hc】
# 7 该cate 平均几天购买一次


import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os
import math
import numpy as np
import gc
from sklearn.model_selection import train_test_split
import xgboost as xgb
from common import *


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   cate基本属性【拥有的商品数，品牌数，有该cate商品的不同店铺数量】
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_cate_feat0():
    dump_path = './cache/basic_cate.pkl'
    if os.path.exists(dump_path):
        cate = pickle.load(open(dump_path, 'rb'))
    else:
        product = pd.read_csv(product_path)
        cate = product.groupby(['cate'])['sku_id', 'brand', 'shop_id'].nunique().reset_index()
        cate.rename(columns={'sku_id': 'num_sku', 'brand': 'num_brand', 'shop_id': 'num_shop'}, inplace=True)
        pickle.dump(cate, open(dump_path, 'wb'))
    return cate


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   对cate下单记录中 不同年龄用户比例
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_cate_feat1(start_date, end_date, actions_all):
    dump_path = './cache/cate_feat1_{}_{}.pkl'.format(start_date, end_date)
    if os.path.exists(dump_path):
        _df = pickle.load(open(dump_path, 'rb'))
    else:
        # actions = get_actions(start_date, end_date)
        # actions = actions_all.copy()
        actions = actions_all[actions_all['type']==2]
        gc.collect()
        user = pd.read_csv(user_path)
        user['age'].fillna(-1.0, inplace=True)
        actions = pd.merge(actions, user[['user_id', 'age']], on='user_id', how='left')
        df  = actions.groupby(['cate', 'age'], as_index=False)['type'].count()  # 品类-属性人群记录数量
        df1 = actions.groupby(['cate'], as_index=False)['action_time'].count()  # 品类记录数量
        df2 = actions.groupby(['age'], as_index=False)['sku_id'].count()
        _df = pd.merge(df, df1, on='cate', how='left')
        _df = pd.merge(_df, df2, on='age', how='left')

        _df['cate_age_ratio'] = _df['type'] / _df['action_time']
        _df['age_cate_ratio'] = _df['type'] / _df['sku_id']
        _df = _df[['cate', 'age', 'cate_age_ratio', 'age_cate_ratio']]
        _df.fillna(0, inplace=True)
        pickle.dump(_df, open(dump_path, 'wb'))
    return _df


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   对cate下单记录中 不同性别用户比例
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_cate_feat2(start_date, end_date, actions_all):
    dump_path = './cache/cate_feat2_{}_{}.pkl'.format(start_date, end_date)
    if os.path.exists(dump_path):
        _df = pickle.load(open(dump_path, 'rb'))
    else:
        # actions = get_actions(start_date, end_date)
        # actions = actions_all.copy()
        actions = actions_all[actions_all['type']==2]
        gc.collect()
        user = pd.read_csv(user_path)
        user['sex'].fillna(2.0, inplace=True)
        actions = pd.merge(actions, user[['user_id', 'sex']], on='user_id', how='left')
        df  = actions.groupby(['cate', 'sex'], as_index=False)['type'].count()  # 品类-属性人群记录数量
        df1 = actions.groupby(['cate'], as_index=False)['action_time'].count()  # 品类记录数量
        df2 = actions.groupby(['sex'], as_index=False)['sku_id'].count()
        _df = pd.merge(df, df1, on='cate', how='left')
        _df = pd.merge(_df, df2, on='sex', how='left')

        _df['cate_sex_ratio'] = _df['type'] / _df['action_time']
        _df['sex_cate_ratio'] = _df['type'] / _df['sku_id']
        _df = _df[['cate', 'sex', 'cate_sex_ratio', 'sex_cate_ratio']]
        _df.fillna(0, inplace=True)
        pickle.dump(_df, open(dump_path, 'wb'))
    return _df


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   对cate下单记录中 不同城市等级用户比例
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_cate_feat3(start_date, end_date, actions_all):
    dump_path = './cache/cate_feat3_{}_{}.pkl'.format(start_date, end_date)
    if os.path.exists(dump_path):
        _df = pickle.load(open(dump_path, 'rb'))
    else:
        # actions = get_actions(start_date, end_date)
        # actions = actions_all.copy()
        actions = actions_all[actions_all['type']==2]
        gc.collect()
        user = pd.read_csv(user_path)
        user['city_level'].fillna(4.0, inplace=True)
        actions = pd.merge(actions, user[['user_id', 'city_level']], on='user_id', how='left')
        df  = actions.groupby(['cate', 'city_level'], as_index=False)['type'].count()  # 品类-属性人群记录数量
        df1 = actions.groupby(['cate'], as_index=False)['action_time'].count()  # 品类记录数量
        df2 = actions.groupby(['city_level'], as_index=False)['sku_id'].count()
        _df = pd.merge(df, df1, on='cate', how='left')
        _df = pd.merge(_df, df2, on='city_level', how='left')

        _df['cate_city_level_ratio'] = _df['type'] / _df['action_time']
        _df['city_level_cate_ratio'] = _df['type'] / _df['sku_id']
        _df = _df[['cate', 'city_level', 'cate_city_level_ratio', 'city_level_cate_ratio']]
        _df.fillna(0, inplace=True)
        pickle.dump(_df, open(dump_path, 'wb'))
    return _df


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   对cate下单记录中 不同城市等级用户比例
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_cate_feat4(start_date, end_date, actions_all):
    dump_path = './cache/cate_feat4_{}_{}.pkl'.format(start_date, end_date)
    if os.path.exists(dump_path):
        _df = pickle.load(open(dump_path, 'rb'))
    else:
        # actions = get_actions(start_date, end_date)
        # actions = actions_all.copy()
        actions = actions_all[actions_all['type']==2]
        gc.collect()
        user = pd.read_csv(user_path)
        user['user_lv_cd'].fillna(4.0, inplace=True)
        actions = pd.merge(actions, user[['user_id', 'user_lv_cd']], on='user_id', how='left')
        df = actions.groupby(['cate', 'user_lv_cd'], as_index=False)['type'].count()  # 品类-属性人群记录数量
        df1 = actions.groupby(['cate'], as_index=False)['action_time'].count()  # 品类记录数量
        df2 = actions.groupby(['user_lv_cd'], as_index=False)['sku_id'].count()
        _df = pd.merge(df, df1, on='cate', how='left')
        _df = pd.merge(_df, df2, on='user_lv_cd', how='left')

        _df['cate_user_lv_cd_ratio'] = _df['type'] / _df['action_time']
        _df['user_lv_cd_cate_ratio'] = _df['type'] / _df['sku_id']
        _df = _df[['cate', 'user_lv_cd', 'cate_user_lv_cd_ratio', 'user_lv_cd_cate_ratio']]
        _df.fillna(0, inplace=True)
        pickle.dump(_df, open(dump_path, 'wb'))
    return _df


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   行为日均值 方差 转化率等
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_cate_feat5(start_date, end_date, actions_all):
    dump_path = './cache/cate_feat5_{}_{}.pkl'.format(start_date, end_date)
    days = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
    days = int(days.days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        features = ['cate',  'cate_action_1_ratio', 'cate_action_3_ratio', 'cate_action_4_ratio']
        
        # actions = get_actions(start_date, end_date)
        actions = actions_all[['cate', 'type']]
        actions = actions[actions['type']<5]
        df = pd.get_dummies(actions['type'], prefix='cate_action')
        cols = ['cate_action_{}'.format(i) for i in range(1, 5)]
        for col in cols:
                if col not in df.columns:
                        df[col] = 0
        actions = pd.concat([actions[['cate']], df], axis=1)
        actions = actions.groupby(['cate'], as_index=False).sum()
        print(actions.columns)
        actions['cate_action_1_ratio'] = (np.log(1 + actions['cate_action_2']) - np.log(1 + actions['cate_action_1']))
        actions['cate_action_3_ratio'] = (np.log(1 + actions['cate_action_2']) - np.log(1 + actions['cate_action_3']))
        actions['cate_action_4_ratio'] = (np.log(1 + actions['cate_action_2']) - np.log(1 + actions['cate_action_4']))

        actions = actions[features]
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['cate'] + ['before_{}_{}'.format(days, col) for col in actions.columns if col not in ['cate']]        
    return actions


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   最近该cate 销量 占所有销量
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_cate_feat6(start_date, end_date, actions_all):
    dump_path = './cache/cate_feat6_{}_{}.pkl'.format(start_date, end_date)
    days = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
    days = int(days.days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        # actions = get_actions(start_date, end_date)
        # actions = actions_all.copy()
        actions = actions_all[actions_all['type'] == 2]
        all_cate_oder_num = actions.shape[0]
        actions = actions.groupby(['cate'], as_index=False)['type'].count()
        actions.columns = ['cate', 'order_num']
        actions['cate_order_num_ratio'] = actions['order_num'] / all_cate_oder_num
        actions = actions[['cate', 'cate_order_num_ratio']]
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['cate'] + ['before_{}_{}'.format(days, col) for col in actions.columns if col not in ['cate']]
    return actions


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   该cate的平均几天购买一次 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_cate_feat7(start_date, end_date, actions_all):
    dump_path = './cache/cate_feat7_{}_{}.pkl'.format(start_date, end_date)
    days = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
    days = int(days.days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        actions = actions_all[['user_id', 'cate', 'action_time']]
        actions['action_time'] = actions['action_time'].map(lambda x:x.split(' ')[0])
        actions = actions.drop_duplicates(['user_id', 'cate', 'action_time'], keep='first')
        actions = actions.groupby(['user_id', 'cate'], as_index=False).count()
        actions['action_time'] = actions['action_time'] / days
        actions = actions.groupby(['cate'], as_index=False)['action_time'].mean()
        actions.columns = ['cate', 'rebuy_rate']
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions


