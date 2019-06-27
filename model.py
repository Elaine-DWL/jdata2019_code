from user import *
from cate import *
from shop import *
from user_cate import *
from cate_shop import *
from user_shop import *
from user_cate_shop import *
from common import *
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
from sklearn.model_selection import StratifiedKFold


# 从Label提取集合中 取出 Label  只要有下单记录的就是正样本    返回指定时间段的正样本
def get_labels(start_date, end_date):
    dump_path = './cache/labels_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        print('获取label区间: ', start_date, end_date)
        actions = get_actions(start_date, end_date)
        actions = actions[actions['type'] == 2]  # 只要下单了  就是正样本
        actions = actions.sort_values(by=['user_id', 'cate', 'action_time'])
        # actions = actions.groupby(['user_id', 'cate'], as_index=False).first()  # 取该cate下 最先下单的那家商店为正样本
        actions['label'] = 1
        actions = actions[['user_id', 'cate', 'shop_id', 'label']]
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions


def make_train_set(train_start_date, train_end_date, fpath=None):
    print("--------构造训练集--------")
    print("特征提取时间段：  ", train_start_date, " --> ", train_end_date)

    # 滑窗区间
    actions = None
    for i in (14, 3, 5, 7, 28):
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')  # 转成字符串
        print("滑窗大小: {}, 范围：{} --> {}".format(i, start_days, train_end_date))
        if actions is None:
            actions = get_action_feat(start_days, train_end_date)  # 最后一天 前i天内的行为统计特征
                   
        else:
            actions = pd.merge(actions, get_action_feat(start_days, train_end_date), 
                                on=['user_id', 'cate', 'shop_id'], how='left')  # 最后一天 前i天内的行为统计特征
    start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=14)
    start_days = start_days.strftime('%Y-%m-%d')  # 转成字符串
    action_all = get_actions(train_start_date, train_end_date)
    actions = pd.merge(actions, get_user_feat0(), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat1(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat2(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat3(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat4(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat5(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat6(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat7(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat8(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat9(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat10(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat12(train_start_date, train_end_date, action_all), on='user_id', how='left')


    actions = pd.merge(actions, get_user_cate_feat1(train_start_date, train_end_date, action_all), on=['user_id', 'cate'], how='left')
    actions = pd.merge(actions, get_user_cate_feat2(train_start_date, train_end_date, action_all), on=['user_id', 'cate'], how='left')
    actions = pd.merge(actions, get_user_cate_feat3(train_start_date, train_end_date, action_all), on=['user_id', 'cate'], how='left')
    actions = pd.merge(actions, get_user_cate_feat4(train_start_date, train_end_date, action_all), on=['user_id', 'cate'], how='left')


    actions = pd.merge(actions, get_cate_feat0(), on='cate', how='left')
    actions = pd.merge(actions, get_cate_feat1(train_start_date, train_end_date, action_all), on=['cate', 'age'], how='left')
    actions = pd.merge(actions, get_cate_feat2(train_start_date, train_end_date, action_all), on=['cate', 'sex'], how='left')
    actions = pd.merge(actions, get_cate_feat3(train_start_date, train_end_date, action_all), on=['cate', 'city_level'], how='left')
    actions = pd.merge(actions, get_cate_feat4(train_start_date, train_end_date, action_all), on=['cate','user_lv_cd'],  how='left')
    actions = pd.merge(actions, get_cate_feat5(train_start_date, train_end_date, action_all), on=['cate'], how='left')
    actions = pd.merge(actions, get_cate_feat6(train_start_date, train_end_date, action_all), on=['cate'], how='left') 

    actions = pd.merge(actions, get_cate_shop_feat1(train_start_date, train_end_date, action_all), on=['cate', 'shop_id'], how='left')
    actions = pd.merge(actions, get_cate_shop_feat2(train_start_date, train_end_date, action_all), on=['cate', 'shop_id'], how='left')

    actions = pd.merge(actions, get_shop_feat0(), on='shop_id', how='left')
    actions = pd.merge(actions, get_shop_feat1(train_start_date, train_end_date, action_all), on=['shop_id', 'age'], how='left')
    actions = pd.merge(actions, get_shop_feat2(train_start_date, train_end_date, action_all), on=['shop_id', 'sex'], how='left')
    actions = pd.merge(actions, get_shop_feat3(train_start_date, train_end_date, action_all), on=['shop_id', 'city_level'], how='left')
    actions = pd.merge(actions, get_shop_feat4(train_start_date, train_end_date, action_all), on=['shop_id', 'user_lv_cd'], how='left')
    actions = pd.merge(actions, get_shop_feat5(train_start_date, train_end_date, action_all), on=['shop_id'], how='left')
    actions = pd.merge(actions, get_shop_feat6(train_start_date, train_end_date, action_all), on='shop_id', how='left')

    actions = pd.merge(actions, get_user_shop_feat1(train_start_date, train_end_date, action_all), on=['user_id', 'shop_id'], how='left')
    actions = pd.merge(actions, get_user_shop_feat2(train_start_date, train_end_date, action_all), on=['user_id', 'shop_id'], how='left')
    actions = pd.merge(actions, get_user_shop_feat3(train_start_date, train_end_date, action_all), on=['user_id', 'shop_id'], how='left')

    actions = reduce_mem_usage(actions)
    print("训练集大小: ", actions.shape)
    actions.fillna(0, inplace=True)
    # actions.to_csv('./file/{}_{}_train_set.csv'.format(train_start_date, train_end_date), index=False)
    # print("已保存到： ", fpath)

    return actions


def make_label(train_start_date, train_end_date, fpath=None):
    label_end_dates = datetime.strptime(train_end_date, '%Y-%m-%d') + timedelta(days=7)
    label_end_dates = label_end_dates.strftime("%Y-%m-%d")
    print("Label提取......")
    labels = get_labels(train_end_date, label_end_dates)
    labels.to_csv(fpath, index=False)
    print("已保存到：", fpath)
    return labels


def make_val_set(train_start_date, train_end_date, fpath):
    print("--------构造验证集--------")
    print("特征提取时间段：  ", train_start_date, " --> ", train_end_date)

    # 滑窗区间
    actions = None
    for i in (14, 3, 5, 7, 28):
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')  # 转成字符串
        print("滑窗大小: {}, 范围：{} --> {}".format(i, start_days, train_end_date))
        if actions is None:
            actions = get_action_feat(start_days, train_end_date)  # 最后一天 前i天内的行为统计特征
                   
        else:
            actions = pd.merge(actions, get_action_feat(start_days, train_end_date), 
                                on=['user_id', 'cate', 'shop_id'], how='left')  # 最后一天 前i天内的行为统计特征
    start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=14)
    start_days = start_days.strftime('%Y-%m-%d')  # 转成字符串
    action_all = get_actions(train_start_date, train_end_date)
    actions = pd.merge(actions, get_user_feat0(), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat1(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat2(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat3(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat4(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat5(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat6(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat7(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat8(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat9(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat10(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat12(train_start_date, train_end_date, action_all), on='user_id', how='left')


    actions = pd.merge(actions, get_user_cate_feat1(train_start_date, train_end_date, action_all), on=['user_id', 'cate'], how='left')
    actions = pd.merge(actions, get_user_cate_feat2(train_start_date, train_end_date, action_all), on=['user_id', 'cate'], how='left')
    actions = pd.merge(actions, get_user_cate_feat3(train_start_date, train_end_date, action_all), on=['user_id', 'cate'], how='left')
    actions = pd.merge(actions, get_user_cate_feat4(train_start_date, train_end_date, action_all), on=['user_id', 'cate'], how='left')


    actions = pd.merge(actions, get_cate_feat0(), on='cate', how='left')
    actions = pd.merge(actions, get_cate_feat1(train_start_date, train_end_date, action_all), on=['cate', 'age'], how='left')
    actions = pd.merge(actions, get_cate_feat2(train_start_date, train_end_date, action_all), on=['cate', 'sex'], how='left')
    actions = pd.merge(actions, get_cate_feat3(train_start_date, train_end_date, action_all), on=['cate', 'city_level'], how='left')
    actions = pd.merge(actions, get_cate_feat4(train_start_date, train_end_date, action_all), on=['cate','user_lv_cd'],  how='left')
    actions = pd.merge(actions, get_cate_feat5(train_start_date, train_end_date, action_all), on=['cate'], how='left')
    actions = pd.merge(actions, get_cate_feat6(train_start_date, train_end_date, action_all), on=['cate'], how='left') 

    actions = pd.merge(actions, get_cate_shop_feat1(train_start_date, train_end_date, action_all), on=['cate', 'shop_id'], how='left')
    actions = pd.merge(actions, get_cate_shop_feat2(train_start_date, train_end_date, action_all), on=['cate', 'shop_id'], how='left')

    actions = pd.merge(actions, get_shop_feat0(), on='shop_id', how='left')
    actions = pd.merge(actions, get_shop_feat1(train_start_date, train_end_date, action_all), on=['shop_id', 'age'], how='left')
    actions = pd.merge(actions, get_shop_feat2(train_start_date, train_end_date, action_all), on=['shop_id', 'sex'], how='left')
    actions = pd.merge(actions, get_shop_feat3(train_start_date, train_end_date, action_all), on=['shop_id', 'city_level'], how='left')
    actions = pd.merge(actions, get_shop_feat4(train_start_date, train_end_date, action_all), on=['shop_id', 'user_lv_cd'], how='left')
    actions = pd.merge(actions, get_shop_feat5(train_start_date, train_end_date, action_all), on=['shop_id'], how='left')
    actions = pd.merge(actions, get_shop_feat6(train_start_date, train_end_date, action_all), on='shop_id', how='left')

    actions = pd.merge(actions, get_user_shop_feat1(train_start_date, train_end_date, action_all), on=['user_id', 'shop_id'], how='left')
    actions = pd.merge(actions, get_user_shop_feat2(train_start_date, train_end_date, action_all), on=['user_id', 'shop_id'], how='left')
    actions = pd.merge(actions, get_user_shop_feat3(train_start_date, train_end_date, action_all), on=['user_id', 'shop_id'], how='left')
    actions = reduce_mem_usage(actions)
    print("验证集大小: ", actions.shape)
    actions.fillna(0, inplace=True)
    actions.to_csv(fpath, index=False)
    print("已保存到： ", fpath)

    return actions


def make_test_set(train_start_date, train_end_date, fpath):
    print("--------构造测试集--------")
    print("特征提取时间段：  ", train_start_date, " --> ", train_end_date)

    # 滑窗区间
    actions = None
    for i in (14, 3, 5, 7, 28):
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')  # 转成字符串
        print("滑窗大小: {}, 范围：{} --> {}".format(i, start_days, train_end_date))
        if actions is None:
            actions = get_action_feat(start_days, train_end_date)  # 最后一天 前i天内的行为统计特征
                   
        else:
            actions = pd.merge(actions, get_action_feat(start_days, train_end_date), 
                                on=['user_id', 'cate', 'shop_id'], how='left')  # 最后一天 前i天内的行为统计特征
    start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=14)
    start_days = start_days.strftime('%Y-%m-%d')  # 转成字符串
    action_all = get_actions(train_start_date, train_end_date)
    actions = pd.merge(actions, get_user_feat0(), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat1(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat2(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat3(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat4(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat5(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat6(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat7(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat8(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat9(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat10(train_start_date, train_end_date, action_all), on='user_id', how='left')
    actions = pd.merge(actions, get_user_feat12(train_start_date, train_end_date, action_all), on='user_id', how='left')


    actions = pd.merge(actions, get_user_cate_feat1(train_start_date, train_end_date, action_all), on=['user_id', 'cate'], how='left')
    actions = pd.merge(actions, get_user_cate_feat2(train_start_date, train_end_date, action_all), on=['user_id', 'cate'], how='left')
    actions = pd.merge(actions, get_user_cate_feat3(train_start_date, train_end_date, action_all), on=['user_id', 'cate'], how='left')
    actions = pd.merge(actions, get_user_cate_feat4(train_start_date, train_end_date, action_all), on=['user_id', 'cate'], how='left')


    actions = pd.merge(actions, get_cate_feat0(), on='cate', how='left')
    actions = pd.merge(actions, get_cate_feat1(train_start_date, train_end_date, action_all), on=['cate', 'age'], how='left')
    actions = pd.merge(actions, get_cate_feat2(train_start_date, train_end_date, action_all), on=['cate', 'sex'], how='left')
    actions = pd.merge(actions, get_cate_feat3(train_start_date, train_end_date, action_all), on=['cate', 'city_level'], how='left')
    actions = pd.merge(actions, get_cate_feat4(train_start_date, train_end_date, action_all), on=['cate','user_lv_cd'],  how='left')
    actions = pd.merge(actions, get_cate_feat5(train_start_date, train_end_date, action_all), on=['cate'], how='left')
    actions = pd.merge(actions, get_cate_feat6(train_start_date, train_end_date, action_all), on=['cate'], how='left') 

    actions = pd.merge(actions, get_cate_shop_feat1(train_start_date, train_end_date, action_all), on=['cate', 'shop_id'], how='left')
    actions = pd.merge(actions, get_cate_shop_feat2(train_start_date, train_end_date, action_all), on=['cate', 'shop_id'], how='left')

    actions = pd.merge(actions, get_shop_feat0(), on='shop_id', how='left')
    actions = pd.merge(actions, get_shop_feat1(train_start_date, train_end_date, action_all), on=['shop_id', 'age'], how='left')
    actions = pd.merge(actions, get_shop_feat2(train_start_date, train_end_date, action_all), on=['shop_id', 'sex'], how='left')
    actions = pd.merge(actions, get_shop_feat3(train_start_date, train_end_date, action_all), on=['shop_id', 'city_level'], how='left')
    actions = pd.merge(actions, get_shop_feat4(train_start_date, train_end_date, action_all), on=['shop_id', 'user_lv_cd'], how='left')
    actions = pd.merge(actions, get_shop_feat5(train_start_date, train_end_date, action_all), on=['shop_id'], how='left')
    actions = pd.merge(actions, get_shop_feat6(train_start_date, train_end_date, action_all), on='shop_id', how='left')

    actions = pd.merge(actions, get_user_shop_feat1(train_start_date, train_end_date, action_all), on=['user_id', 'shop_id'], how='left')
    actions = pd.merge(actions, get_user_shop_feat2(train_start_date, train_end_date, action_all), on=['user_id', 'shop_id'], how='left')
    actions = pd.merge(actions, get_user_shop_feat3(train_start_date, train_end_date, action_all), on=['user_id', 'shop_id'], how='left')

    actions = reduce_mem_usage(actions)
    print("测试集大小: ", actions.shape)
    actions.fillna(0, inplace=True)
    actions.to_csv(fpath, index=False)
    print("已保存到： ", fpath)

    return actions


# def score(pred, label):
#     # 计算 [user_id  cate] 的precision和 recall
#     # 预测出的 user_id cate对
#     print('pred: ', pred.shape[0])
#     print('label: ', label.shape[0])
#     pred_all_user_cate_pair = pred['user_id'].map(str) + '-' + pred['cate'].map(str)
#     pred_all_user_cate_pair = np.array(pred_all_user_cate_pair)
#     # 真实的  user_id cate 对
#     label_all_user_cate_pair = label['user_id'].map(str) + '-' + label['cate'].map(str)
#     label_all_user_cate_pair = np.array(label_all_user_cate_pair)
#     TP_1 = len(set(pred_all_user_cate_pair) & set(label_all_user_cate_pair))
#     recall_1 = TP_1 / label.shape[0]
#     precision_1 = TP_1 / pred.shape[0]
#     F1_1 = 3 * precision_1 * recall_1 / (2 * recall_1 + precision_1)
#     print("TP_1 :          ", TP_1)
#     print("Precision_1 :   ", precision_1)
#     print("Recall_1 :      ", recall_1)
#     print("F1_1 :          ", F1_1)

#     pred_all_user_cate_shop_pair = pred['user_id'].map(str) + '-' + pred['cate'].map(str) + '-' + pred['shop_id'].map(
#         str)
#     pred_all_user_cate_shop_pair = np.array(pred_all_user_cate_shop_pair)
#     # 真实的  user_id cate 对
#     label_all_user_cate_shop_pair = label['user_id'].map(str) + '-' + label['cate'].map(str) + '-' + label[
#         'shop_id'].map(str)
#     label_all_user_cate_shop_pair = np.array(label_all_user_cate_shop_pair)

#     TP_2 = len(set(pred_all_user_cate_shop_pair) & set(label_all_user_cate_shop_pair))
#     recall_2 = TP_2 / label.shape[0]
#     precision_2 = TP_2 / pred.shape[0]
#     F1_2 = 5 * precision_2 * recall_2 / (2 * recall_2 + 3 * precision_2)
#     print("TP_2 :          ", TP_2)
#     print("Precision_2 :   ", precision_2)
#     print("Recall_2 :      ", recall_2)
#     print("F1_2 :          ", F1_2)
#     return 0.4 * F1_1 + 0.6 * F1_2


# def xgb_train(params, train_set, model_path, zero_cols=[]):
#     train_x = pd.read_csv(train_set)
#     print("训练集加载完成!")
#     # train_x = train_x[train_x['cate'] > 0] # 删除对不存在与 product中的商品的行为记录
#     # print(gc.collect())
#     # labels = pd.read_csv(train_label)
#     # print("训练集标签加载完成!")
#     # train_x = pd.merge(train_x, labels, on=['user_id', 'cate', 'shop_id'], how='left')  # 这是错的！！！不能merge
#     # train_x = pd.concat([train_x, labels], axis=1)
#     # del labels
#     # gc.collect()
#     train_x.fillna(0, inplace=True)
#     print("训练集正负样本比例： ", train_x['label'].value_counts())
#     # --------------负样本采样-----------------------------------------------
#     positive = train_x[train_x['label'] > 0]  # 原始正样本作为正样本
#     train_x = train_x[train_x['label'] < 1]  # 原始负样本
#     gc.collect()
#     # 对负样本进行筛选   选出质量好的负样本   
#     # 最可能购买 但是最后没有购买 的 负样本
#     train_x = train_x[train_x['user_action_2_last_diff'] < 15]  # 最近两周有购买行为  但是最后没买
#     train_x = train_x[train_x['user_action_1_last_diff'] < 8]  # 最近xxx有浏览 但是最后没买
#     #  计算cate相似度矩阵  用协同过滤 来做    待做
#     gc.collect()
#     # 对负样本进行采样
#     train_x = train_x.sample(n = 10*len(positive), random_state=0)
#     train_x = pd.concat([positive, train_x], ignore_index=True)
#     del positive
#     gc.collect()
#     # -------------------------------------------------------------------------
#     print("训练集正负样本比例： ", train_x['label'].value_counts())
#     # val_x = pd.read_csv(val_set)
#     train_y = train_x['label']
#     # val_y =  val_x['label']

#     to_drop = ['user_id', 'cate', 'shop_id', 'label', 'age', 'sex', 'city_level', 'user_lv_cd'] + zero_cols
#     train_x.drop(to_drop, axis=1, inplace=True)
#     # val_x.drop(to_drop, axis=1, inplace=True)
#     feature_names = train_x.columns

#     print("train_x.shape :", train_x.shape)
#     # print("val_x.shape :", val_x.shape)

#     X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y,
#                                                           test_size=0.2, random_state=0)
#     dtrain = xgb.DMatrix(X_train.values, label=y_train.values, feature_names=feature_names)
#     dvalid = xgb.DMatrix(X_valid.values, label=y_valid.values, feature_names=feature_names)

#     # dtrain = xgb.DMatrix(train_x.values, label=train_y.values, feature_names=feature_names)
#     # dvalid = xgb.DMatrix(val_x.values, label=val_y.values, feature_names=feature_names)
#     evallist = [(dtrain, 'train'), (dvalid, 'eval')]
#     num_rounds = 200
#     model = xgb.train(dtrain=dtrain, params=params, evals=evallist, num_boost_round=num_rounds,
#                       verbose_eval=1, early_stopping_rounds=10)

#     model.save_model(model_path)
#     feats = model.get_fscore().keys()
#     zero_col = [col for col in feature_names if col not in feats]
#     return model, zero_col


def Recall(df_true_pred, threshold, flag):
    """
        df_true_pred : 'user_id', 'cate', 'shop_id', 'label', 'pred_prob'
            flag : 'user_cate' or 'user_cate_shop'
            Threshold = 0.5
            Recall = TP / (TP + FP)
    """
    temp_ = df_true_pred[df_true_pred['pred_prob']>=threshold]
    if flag == 'user_cate':
        temp_ = temp_.drop_duplicates(['user_id', 'cate'])
        recall = np.sum(temp_['label']) * 1.0 / np.sum(df_true_pred['label'])
    elif flag == 'user_cate_shop':
        recall = np.sum(temp_['label']) * 1.0 / np.sum(df_true_pred['label'])
    else:
        recall = -1
    return recall

def Precision(df_true_pred, threshold, flag):
    """
        df_true_pred : 'user_id', 'cate', 'shop_id', 'label', 'pred_prob'
            flag : 'user_cate' or 'user_cate_shop'
            Threshold 
            Precision = TP / (TP + TN)
    """
    temp_ = df_true_pred[df_true_pred['pred_prob']>=threshold]
    if flag == 'user_cate':
        temp_ = temp_.drop_duplicates(['user_id', 'cate'])
        precision = np.sum(temp_['label']) * 1.0 / np.size(df_true_pred)
    elif flag == 'user_cate_shop':
        precision = np.sum(temp_['label']) * 1.0 / np.size(df_true_pred)
    else:
        precision = -1
    return precision

def get_metrics(df_true_pred, threshold):
    """
        df_true_pred : 'user_id', 'cate', 'shop_id', 'label', 'pred_prob'
        Threshold = 0.5
    """ 

    # 用户-品类
    R1_1 = Recall(df_true_pred, threshold, flag='user_cate') 
    P1_1 = Precision(df_true_pred, threshold, flag='user_cate')
    F1_1 = 3 * R1_1 * P1_1 / (2 * R1_1 + P1_1)

    # 用户-品类-店铺
    R1_2 = Recall(df_true_pred, threshold, flag='user_cate_shop') 
    P1_2 = Precision(df_true_pred, threshold, flag='user_cate_shop')
    F1_2 = 5 * R1_2 * P1_2 / (2 * R1_2 + 3 * P1_2)

    # 总分
    score = 0.4 * F1_1 + 0.6 * F1_2

    print("R1_1: ", R1_1)
    print("P1_1: ", P1_1)
    print("F1_1: ", F1_1)
    print("-------------------------")
    print("R1_2: ", R1_2)
    print("P1_2: ", P1_2)
    print("F1_2: ", F1_2)
    print("-------------------------")
    print("score: ", score)
    return score

def xgb_train(params, train_set, test_set, test_label, model_path, zero_cols=[]):
    train_x = pd.read_csv(train_set)
    print("训练集加载完成!")
    train_x.fillna(0, inplace=True)
    print("训练集正负样本比例： ", train_x['label'].value_counts())
    # --------------负样本采样-----------------------------------------------
    positive = train_x[train_x['label'] > 0]  # 原始正样本作为正样本
    train_x = train_x[train_x['label'] < 1]  # 原始负样本
    gc.collect()
    # 对负样本进行筛选   选出质量好的负样本   
    # 最可能购买 但是最后没有购买 的 负样本
    train_x = train_x[train_x['user_action_2_last_diff'] < 8]  # 最近两周有购买行为  但是最后没买
    train_x = train_x[train_x['user_action_1_last_diff'] < 3]  # 最近xxx有浏览 但是最后没买
    #  计算cate相似度矩阵  用协同过滤 来做    待做
    gc.collect()
    # 对负样本进行采样
    print(train_x.shape, positive.shape)
    # if(train_x.shape[0] > len(positive)):
    #     train_x = train_x.sample(n = len(positive), random_state=0)
    train_x = pd.concat([positive, train_x], ignore_index=True)
    # del positive
    gc.collect()
    # -------------------------------------------------------------------------
    print("下采样后训练集正负样本比例： \n", train_x['label'].value_counts())
    train_y = train_x['label']
    to_drop = ['user_id', 'cate', 'shop_id', 'label', 'age', 'sex', 'city_level', 'user_lv_cd'] + zero_cols
    train_x.drop(to_drop, axis=1, inplace=True)
    feature_names = train_x.columns
    print("train_x.shape :", train_x.shape)

    test_x = pd.read_csv(test_set)  # 其实是线下的验证集
    labels = pd.read_csv(test_label)
    test_x = pd.merge(test_x, labels, on=['user_id', 'cate', 'shop_id'], how='left')
    test_x.fillna(0, inplace=True)
    del labels
    gc.collect()
    user_index = test_x[['user_id', 'cate', 'shop_id', 'label']]
    to_drop = ['user_id', 'cate', 'shop_id', 'age', 'sex', 'city_level', 'user_lv_cd', 'label'] + zero_cols
    test_x.drop(to_drop, axis=1, inplace=True)
    gc.collect()
    # k折交叉验证
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2019)
    oof_train = np.zeros((train_x.shape[0],))
    oof_test = np.zeros((test_x.shape[0], n_folds))
    # dtest = xgb.DMatrix(test_x.values, feature_names=feature_names)
    feature_names = test_x.columns
    dtest = xgb.DMatrix(test_x.values, user_index['label'].values)
    models = []
    i = 0
    for train_index, val_index in skf.split(train_x, train_y):
        X_train = train_x.loc[train_index, :]
        y_train = train_y.loc[train_index]
        X_val = train_x.loc[val_index, :]
        y_val = train_y.loc[val_index]

        dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
        dvalid = xgb.DMatrix(X_val.values, label=y_val.values)

        evalist = [(dtrain, 'train'), (dvalid, 'eval'), (dtest, 'valid')]
        num_rounds = 4000

        model = xgb.train(dtrain=dtrain, params=params, evals=evalist, num_boost_round=num_rounds,
                      verbose_eval=1, early_stopping_rounds=50)
        # feature_importance = pd.DataFrame({'feature': feature_names})

        # fi_feature = list(model.get_fscore().keys)
        # fi_importance = list(model.get_fscore().values)
        # tdf = pd.DataFrame({'feature': fi_feature, 'importance': fi_importance})
        # feature_importance = pd.merge(feature_importance, tdf, on='feature', how='left')
        # feature_importance.fillna(0, inplace=True)
        # feature_importance.to_csv('model_{}_importance.csv', index=False)

        oof_train[val_index] = model.predict(xgb.DMatrix(X_val.values),
                                  ntree_limit=model.best_ntree_limit)
        oof_test[:, i] = model.predict(dtest, ntree_limit=model.best_ntree_limit) 
        model.save_model('model_{}.model'.format(i))
        models.append(model)
        i = i+1
    user_index['pred_prob'] = oof_test.mean(axis=1)
    get_metrics(user_index, 0.4)
    # pickle.dump(models, open('model.pickle', rb))
    return models, feature_names


def validate(model, val_set, val_label, verbose=True, zero_cols=[]):
    val = pd.read_csv(val_set)
    labels = pd.read_csv(val_label)
    val.drop(zero_cols, axis=1, inplace=True)
    val = pd.merge(val, labels, on=['user_id', 'cate', 'shop_id'], how='left')
    val.fillna(0, inplace=True)

    val_y = val['label']
    user_index = val[val['label'] > 0]
    user_index = user_index[['user_id', 'cate', 'shop_id']]  # 验证集中所有正样本的pairs

    to_drop = ['user_id', 'cate', 'shop_id', 'label', 'age', 'sex', 'city_level', 'user_lv_cd']
    cols = [col for col in val.columns if col not in to_drop]

    print("val_x.shape : ", val[cols].shape)
    print("val_y.shape : ", val_y.shape)
    print("user_index.shape : ", user_index.shape)

    dval = xgb.DMatrix(val[cols])
    val_preds = model.predict(dval)
    val['preds'] = val_preds
    val = val[['user_id', 'cate', 'shop_id', 'label', 'preds']]
    # pos_num = val[val['label']>0].shape[0]  # 真实的正样本数量
    pos_num = 10000
    if verbose:  # 先去重  再取 100000   最后等于 100000
        val = val.sort_values(by=['user_id', 'cate', 'preds'])
        df = val.groupby(['user_id', 'cate'], as_index=False).last()
        df = df[:pos_num]
    else:  # 取前100000 个   再来去重  最后少于100000
        val = val.sort_values(by=['preds'], ascending=False)
        df = val[:pos_num]
        df = df.sort_values(by=['user_id', 'cate', 'preds'])
        df = df.groupby(['user_id', 'cate'], as_index=False).last()
    print("检查【user_id, cate】重复数量：", df.duplicated(['user_id', 'cate']).sum())
    print("验证集得分：", score(df[['user_id', 'cate', 'shop_id']], user_index))


def show_feature_importance(model):
    fi_df = pd.DataFrame({
        'cols': list(model.get_fscore().keys()),
        'importance': list(model.get_fscore().values())
    }).sort_values(by='importance', ascending=False)
    print(fi_df.head(10))
    print(fi_df.tail(10))


def make_submission(models, test_set, fpath, verbose=True, zero_cols=[]):
    # reader = pd.read_csv(test_set, chunksize=100000)
    # i = 0
    # for read in reader:
    #     if i==0:
    #         test_x = read
    #         test_x = test_x[test_x['user_action_2_last_diff'] < 15]  # 最近两周有购买行为  但是最后没买
    #         test_x = test_x[test_x['user_action_1_last_diff'] < 7]  # 最近xxx有浏览 但是最后没买 
    #         test_x.drop(zero_cols, axis=1, inplace=True)
    #     else:
    #         t = read
    #         t = t[t['user_action_2_last_diff'] < 15]  # 最近两周有购买行为  但是最后没买
    #         t = t[t['user_action_1_last_diff'] < 7]  # 最近xxx有浏览 但是最后没买 
    #         t.drop(zero_cols, axis=1, inplace=True)
    #         test_x = pd.concat([test_x, t], axis=0)
    #     i = i+1
    #     print(i)

    # del reader, i
    # gc.collect()
    test_x = pd.read_csv(test_set)
    # -----------------------------删除最近半个月没有下单记录且最近一周没有浏览行为的----------
    # test_x = test_x[test_x['user_action_2_last_diff'] < 15]  # 最近两周有购买行为  但是最后没买
    # test_x = test_x[test_x['user_action_1_last_diff'] < 7]  # 最近xxx有浏览 但是最后没买 
    # print(gc.collect())
    # -----------------------------删除最近半个月没有下单记录且最近一周没有浏览行为的----------

    test = test_x[['user_id', 'cate', 'shop_id']]
    to_drop = ['user_id', 'cate', 'shop_id', 'age', 'sex', 'city_level', 'user_lv_cd']
    cols = [col for col in test_x.columns if col not in to_drop]
    test_x.drop(to_drop, axis=1, inplace=True)
    print("test_x.shape : ", test_x[cols].shape)
    test_preds = np.zeros((test_x.shape[0], len(models)))

    dtest = xgb.DMatrix(test_x[cols], feature_names=cols)
    i = 0
    for model in models:
        test_preds[:, i] = model.predict(dtest, ntree_limit=model.best_ntree_limit)
        i = i+1

    test['preds'] = test_preds.mean(axis=1)
    
    pos_num = 10000
    if verbose:  # 先去重  再取 100000   最后等于 100000
        test = test.sort_values(by=['user_id', 'cate', 'preds'])
        df = test.groupby(['user_id', 'cate'], as_index=False).last()
        df = df[:pos_num]
    else:  # 取前100000 个   再来去重  最后少于100000
        test = test.sort_values(by=['preds'], ascending=False)
        df = test[:pos_num]
        df = df.sort_values(by=['user_id', 'cate', 'preds'])
        df = df.groupby(['user_id', 'cate'], as_index=False).last()
    df = df[['user_id', 'cate', 'shop_id']]
    print("检查【user_id, cate】重复数量：", df.duplicated(['user_id', 'cate']).sum())

    # 类型转换
    df['user_id'] = df['user_id'].astype(int)
    df['cate'] = df['cate'].astype(int)
    df['shop_id'] = df['shop_id'].astype(int)
    print("预测文件大小： ", df.shape)
    # 生成提交文件
    df.to_csv(fpath, index=False)
    print("预测文件生成完成！")


# 根据起始日期   滑窗构建多组训练集  
def make_n_train_set(train_start_date, setNums, train_set_path):  # 给定训练开始时间  训练集数量
    train_actions = None
    # 滑窗,构造多组训练集/验证集
    for i in range(setNums):  # 要构造的训练集数量
        print("构造第 {} 组训练集...".format(i))
        train_end_date = datetime.strptime(train_start_date, '%Y-%m-%d') + timedelta(days=30)  # 这里时间间隔是1天
        train_end_date = train_end_date.strftime('%Y-%m-%d')
        label_date_end = datetime.strptime(train_end_date, '%Y-%m-%d') + timedelta(days=7)  # 这里时间间隔是1天
        label_date_end = label_date_end.strftime('%Y-%m-%d')
        if train_actions is None:
            train_actions = make_train_set(train_start_date, train_end_date)
            train_label = get_labels(train_end_date, label_date_end)
            train_actions = pd.merge(train_actions, train_label, on=['user_id', 'cate', 'shop_id'], how='left')
            # 采样
            train_actions.fillna(0 , inplace=True)
            action_postive = train_actions[train_actions['label'] == 1]
            action_negative = train_actions[train_actions['label'] == 0]
            print("正： 负 ", action_postive.shape[0], ": ", action_negative.shape[0])
            # neg_len = len(action_postive) * 10
            # action_negative = action_negative.sample(n=neg_len)

            train_actions = pd.concat([action_postive, action_negative], ignore_index=True) 
            del action_postive, action_negative
            gc.collect()
        else:
            train_x = make_train_set(train_start_date, train_end_date)
            train_y = get_labels(train_end_date, label_date_end)
            train_x = pd.merge(train_x, train_y, on=['user_id', 'cate', 'shop_id'], how='left')
            # 采样
            train_x.fillna(0 , inplace=True)
            action_postive = train_x[train_x['label'] == 1]
            action_negative = train_x[train_x['label'] == 0]
            print("正： 负 ", action_postive.shape[0], ": ", action_negative.shape[0])
            # neg_len = len(action_postive) * 10
            # action_negative = action_negative.sample(n=neg_len)
            train_actions = pd.concat([train_actions, action_postive, action_negative], ignore_index=True)
            del action_postive, train_x, train_y, action_negative
            gc.collect()
        # 接下来每次移动一天

        train_start_date = datetime.strptime(train_start_date, '%Y-%m-%d') + timedelta(days=7)  # 这里时间间隔是1天
        train_start_date = train_start_date.strftime('%Y-%m-%d')
        print("round {0}/{1} over!".format(i+1, setNums))
    print("原始训练集大小: ", train_actions.shape)
    train_actions.to_csv(train_set_path, index=False)
