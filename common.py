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

# action_path = "../data/jdata_action.csv"
action_path = "../data/action.csv"
comment_path = "../data/jdata_comment.csv"
product_path = "../data/jdata_product.csv"
user_path = "../data/jdata_user.csv"
shop_path = "../data/jdata_shop.csv"


# 节约内存的函数
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        if col in ['user_id', 'cate', 'shop_id']:
            continue
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                else:
                        df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                    start_mem - end_mem) / start_mem))
    return df


# 返回指定时间段的原始action数据
def get_actions(start_date, end_date):
    dump_path = './cache/all_action_{}_{}.pkl'.format(start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        actions = pd.read_csv(action_path)
        actions = actions[(actions.action_time >= start_date) & (actions.action_time < end_date)]

#         pickle.dump(actions, open(dump_path, 'wb'))
    return actions


def get_action_feat(start_date, end_date):  # i是时间间隔
    dump_path = './cache/action_feat_{}_{}.pkl'.format(start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        actions = get_actions(start_date, end_date)
        days = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
        i = int(days.days)
        actions = actions[['user_id', 'cate', 'shop_id', 'type']]
        # 不同时间累积的行为计数（3,5,7,10,15,21,30）
        actions = actions[actions['type']<5]
        df = pd.get_dummies(actions['type'], prefix='action_before_%s' % i)
        cols = ['action_before_{}_{}'.format(i, j) for j in range(1, 5)]
        for col in cols:
            if col not in df.columns:
                    df[col] = 0
        before_date = 'action_before_%s' % i
        actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
        # 分组统计，用户-类别-商品,不同用户对不同类别下商品的行为计数
        actions = actions.groupby(['user_id', 'cate', 'shop_id'], as_index=False).sum()
        # 分组统计，用户-类别，不同用户对不同商品类别的行为计数
        user_cate = actions.groupby(['user_id', 'cate'], as_index=False).sum()
        del user_cate['shop_id']
        del user_cate['type']
        actions = pd.merge(actions, user_cate, how='left', on=['user_id', 'cate'])

        # 本类别下其他商品点击量
        # 前述两种分组含有相同名称的不同行为的计数，系统会自动针对名称调整添加后缀,x,y，
        # 所以这里作差统计的是同一类别下其他商品的行为计数
        actions[before_date + '_1_y'] = actions[before_date + '_1_y'] - actions[before_date + '_1_x']
        actions[before_date + '_2_y'] = actions[before_date + '_2_y'] - actions[before_date + '_2_x']
        actions[before_date + '_3_y'] = actions[before_date + '_3_y'] - actions[before_date + '_3_x']
        actions[before_date + '_4_y'] = actions[before_date + '_4_y'] - actions[before_date + '_4_x']

        actions[before_date + 'minus_mean_1'] = actions[before_date + '_1_x'] - (actions[before_date + '_1_x'] / i)
        actions[before_date + 'minus_mean_2'] = actions[before_date + '_2_x'] - (actions[before_date + '_2_x'] / i)
        actions[before_date + 'minus_mean_3'] = actions[before_date + '_3_x'] - (actions[before_date + '_3_x'] / i)
        actions[before_date + 'minus_mean_4'] = actions[before_date + '_4_x'] - (actions[before_date + '_4_x'] / i)
        
        # ----------------------------------此处也可以对user_id shop_id 进行 group 然后进行相关操作

        del actions['type']
        del_cols = [before_date + '_{}_x'.format(i) for i in range(1, 5)]
        actions.drop(del_cols, axis=1, inplace=True)
#         pickle.dump(actions, open(dump_path, 'wb'))

    return actions