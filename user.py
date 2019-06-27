# 0 用户基本属性
# 1 用户对在时间段内行为i的天数 / 时间段大小           [hc]
# 2 用户最后一次行为i距离end_date 的时间【天】
# 3 用户行为i的记录数 均值 方差  和 下单转化率         [hc]
# 4 用户订单数、用户购买了几周、用户连续购买了几周
# 5 用户浏览数、用户浏览了几周、用户连续浏览了几周
# 6 用户浏览前访问天数
# 7 用户关注前浏览天数
# 8 用户购买前关注天数
# 9 用户四种行为 分别的平均访问时间间隔(天)   所有相邻行为间隔的平均值
# 10 用户四种行为的频率 = 行为的次数 / （最后一次时间-最早的时间）
# 11 用户对品类的重购率
# 12 用户最后一次行为的次数 并且进行归一化

from common import *
from sklearn import preprocessing

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   用户表中构造的用户属性特征【年龄，性别，会员等级，所在城市等级】
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_user_feat0():
    dump_path = './cache/basic_user.pkl'
    if os.path.exists(dump_path):
        user = pickle.load(open(dump_path, 'rb'))
    else:
        user = pd.read_csv(user_path)
        # 缺失值填补
        user['age'].fillna(-1.0, inplace=True)
        user['sex'].fillna(2.0, inplace=True)
        user['city_level'].fillna(4.0, inplace=True)
        # user['province'].fillna(-1.0, inplace=True)
        # user['city'].fillna(-1.0, inplace=True)
        # user['county'].fillna(-1.0, inplace=True)
        age_df = pd.get_dummies(user["age"], prefix="age")
        sex_df = pd.get_dummies(user["sex"], prefix="sex")
        city_df = pd.get_dummies(user["city_level"], prefix="city_level")
        user_lv_df = pd.get_dummies(user["user_lv_cd"], prefix="user_lv_cd")

        user = pd.concat([user[['user_id', 'age', 'sex', 'city_level', 'user_lv_cd']], age_df, sex_df, user_lv_df, city_df], axis=1)
#         pickle.dump(user, open(dump_path, 'wb'))
    return user

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   用户浏览/下单/评论/关注 平均时间间隔 (天)【**************】
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_user_feat1(start_date, end_date, actions_all):
    dump_path = './cache/user_feat1_{}_{}.pkl'.format(start_date, end_date)
    days = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
    days = int(days.days)
    if os.path.exists(dump_path):
            _df = pickle.load(open(dump_path, 'rb'))
    else:
        # actions = get_actions(start_date, end_date)
        actions = actions_all.copy()
        _df = None
        for i in range(1, 5):
            df = actions[actions['type']==i]
            df = df[['user_id', 'action_time']]
            df['action_time'] = pd.to_datetime(df['action_time']).apply(lambda x:x.date())
            df = df.drop_duplicates(['user_id', 'action_time'], keep='first')
            df = df.sort_values(by=['user_id', 'action_time'])
            df = df.groupby(['user_id'], as_index=False)['action_time'].count()  # 该用户有多少天有行为i 
            df['avg_action_{}_day'.format(i)] = df['action_time'] / days
            df = df[['user_id', 'avg_action_{}_day'.format(i)]]
            if i == 1:
                _df = df
            else:
                _df = pd.merge(_df, df, on='user_id', how='outer')
        _df.fillna(0, inplace=True)
#         pickle.dump(_df, open(dump_path, 'wb'))
    _df.columns = ['user_id'] + ["before_{}_{}".format(days, col) for col in _df.columns if col not in ['user_id']]
    return  _df


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   用户end_date 前最后一次[1,2,3,4]行为时间距离end_date有多久
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_user_feat2(start_date, end_date, actions_all):
    dump_path = './cache/user_feat2_{}_{}.pkl'.format(start_date, end_date)
    if os.path.exists(dump_path):
            _df = pickle.load(open(dump_path, 'rb'))
    else:
        # actions = get_actions(start_date, end_date)
        actions = actions_all.copy()
        actions['action_time'] = pd.to_datetime(actions['action_time'])
        _df = None
        for i in range(1, 5):
            df = actions[actions['type']==i]
            df = df.sort_values(by=['user_id', 'action_time'])
            df = df.groupby(['user_id'], as_index=False)['action_time'].last()
            df['end_date'] = datetime.strptime(end_date, '%Y-%m-%d')
            df['user_action_{}_last_diff'.format(i)] = df['end_date'] - df['action_time']
            df['user_action_{}_last_diff'.format(i)] = df['user_action_{}_last_diff'.format(i)].map(lambda x:int(x.days))
            df = df[['user_id', 'user_action_{}_last_diff'.format(i)]]
            gc.collect()
            if i == 1:
                _df = df
            else:
                _df = pd.merge(_df, df, on='user_id', how='outer')
        _df.fillna(30, inplace=True)
#         pickle.dump(_df, open(dump_path, 'wb'))
    # _df.columns = ['user_id'] + ['u_feat2_'+ str(i) for i in range(1, _df.shape[1])]
    return _df


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   用户行为i的记录数 均值 方差  和 下单转化率
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_user_feat3(start_date, end_date, actions_all):
    dump_path = './cache/user_feat3_{}_{}.pkl'.format(start_date, end_date)
    days = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
    days = int(days.days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        features = ['user_id', 'type_1_ratio', 'type_3_ratio', 'type_4_ratio']
        
        # actions = get_actions(start_date, end_date)
        # actions = actions_all.copy()
        actions = actions_all[actions_all['type'] < 5]
        df = pd.get_dummies(actions['type'], prefix='type')
        cols = ['type_{}'.format(i) for i in range(1, 5)]
        for col in cols:
            if col not in df.columns:
                    df[col] = 0
        actions = pd.concat([actions[['user_id']], df], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        # 转化率
        actions['type_1_ratio'] = np.log(actions['type_2']) - np.log(1 + actions['type_1'])
        actions['type_3_ratio'] = np.log(actions['type_2']) - np.log(1 + actions['type_2'])
        actions['type_4_ratio'] = np.log(actions['type_2']) - np.log(1 + actions['type_3'])

        actions.fillna(0, inplace=True)
        actions = actions[features]
#         pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id'] + ["before_{}_{}".format(days, col) for 
                                        col in actions.columns if col not in ['user_id']]
    return actions


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   用户订单数、用户购买了几周、用户连续购买了几周/    
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_user_feat4(start_date, end_date, actions_all):
    dump_path = './cache/user_feat4_{}_{}.pkl'.format(start_date, end_date)
    days = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
    days = int(days.days)  # 时间间隔的大小
    if os.path.exists(dump_path):
        df1 = pickle.load(open(dump_path, 'rb'))
    else:
        # actions = actions_all.copy()
        actions = actions_all[['user_id', 'action_time', 'type']]
        gc.collect()
        actions['action_time'] = actions['action_time'].apply(lambda x:x.split()[0])
        actions = actions[actions['type']==2]
        # 用户订单数
        df1 = actions.groupby(['user_id'], as_index=False)['type'].count() # 用户在该时间段内的订单数
        df1.columns = ['user_id', 'order_num']
        # 用户购买了几周
        week1_end = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=7)
        week1_end = week1_end.strftime('%Y-%m-%d')
        week2_end = datetime.strptime(week1_end, '%Y-%m-%d') + timedelta(days=7)
        week2_end = week2_end.strftime('%Y-%m-%d')
        week3_end = datetime.strptime(week2_end, '%Y-%m-%d') + timedelta(days=7)
        week3_end = week3_end.strftime('%Y-%m-%d')
        week4_end = datetime.strptime(week3_end, '%Y-%m-%d') + timedelta(days=7)
        week4_end = week4_end.strftime('%Y-%m-%d')

        week1 = actions[(actions['action_time']>=start_date) & (actions['action_time']<week1_end)] # 第一周的所有订单
        week1 = week1.drop_duplicates(['user_id'], keep='first')  # 第一周下单了的所有用户
        week1 = week1[['user_id']]
        week1['week1'] = 1
        gc.collect()

        week2 = actions[(actions['action_time']>=week1_end) & (actions['action_time']<week2_end)] # 第二周的所有订单
        week2 = week2.drop_duplicates(['user_id'], keep='first')  # 第二周下单了的所有用户
        week2 = week2[['user_id']]
        week2['week2'] = 1
        gc.collect()

        week3 = actions[(actions['action_time']>=week2_end) & (actions['action_time']<week3_end)] # 第二周的所有订单
        week3 = week3.drop_duplicates(['user_id'], keep='first')  # 第二周下单了的所有用户
        week3 = week3[['user_id']]
        week3['week3'] = 1
        gc.collect()

        week4 = actions[(actions['action_time']>=week3_end) & (actions['action_time']<week4_end)] # 第二周的所有订单
        week4 = week4.drop_duplicates(['user_id'], keep='first')  # 第二周下单了的所有用户
        week4 = week4[['user_id']]
        week4['week4'] = 1
        gc.collect()

        week = pd.merge(week1, week2, on=['user_id'], how='outer')
        week = pd.merge(week, week3, on=['user_id'], how='outer')
        week = pd.merge(week, week4, on=['user_id'], how='outer')
        del week1, week2, week3, week4
        gc.collect()
        # 用户连续购买了几周
        week.fillna(0, inplace=True)
        week['week_buy_num'] = week['week1']+week['week2']+week['week3']+week['week4']  # 用户购买了几周
        week['week_buy_lianxu'] = week['week4']
        week.loc[week['week_buy_lianxu']==1, 'week_buy_lianxu'] = week[week['week_buy_lianxu']==1]['week_buy_lianxu'] +\
                                                        week[week['week_buy_lianxu']==1]['week3']
        week.loc[week['week_buy_lianxu']==2, 'week_buy_lianxu'] = week[week['week_buy_lianxu']==2]['week_buy_lianxu'] +\
                                                        week[week['week_buy_lianxu']==2]['week2']                        
        week.loc[week['week_buy_lianxu']==3, 'week_buy_lianxu'] = week[week['week_buy_lianxu']==3]['week_buy_lianxu'] +\
                                                        week[week['week_buy_lianxu']==3]['week1']
        week = week[['user_id', 'week_buy_num', 'week_buy_lianxu']]
        
        df1 = pd.merge(df1, week, on='user_id', how='left')
        df1 = df1[['user_id', 'order_num', 'week_buy_num', 'week_buy_lianxu']]
#         pickle.dump(df1, open(dump_path, 'wb'))
    return df1


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   用户浏览的天数、用户浏览了几周、用户连续浏览了几周    
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_user_feat5(start_date, end_date, actions_all):
    dump_path = './cache/user_feat5_{}_{}.pkl'.format(start_date, end_date)
    days = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
    days = int(days.days)  # 时间间隔的大小
    if os.path.exists(dump_path):
        df1 = pickle.load(open(dump_path, 'rb'))
    else:
        # actions = actions_all.copy()
        actions = actions_all[['user_id', 'type', 'action_time']]
        actions = actions[actions['type']==1]  # 所有的浏览记录
        actions['action_time'] = actions['action_time'].apply(lambda x:x.split()[0])
        # 用户浏览的天数
        actions = actions.drop_duplicates(['user_id', 'action_time'], keep='first')  # 用户每天保留一条记录
        df1 = actions.groupby(['user_id'], as_index=False)['type'].count() # 用户浏览的天数
        df1.columns = ['user_id', 'read_num']
        # 用户浏览了几周
        week1_end = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=7)
        week1_end = week1_end.strftime('%Y-%m-%d')
        week2_end = datetime.strptime(week1_end, '%Y-%m-%d') + timedelta(days=7)
        week2_end = week2_end.strftime('%Y-%m-%d')
        week3_end = datetime.strptime(week2_end, '%Y-%m-%d') + timedelta(days=7)
        week3_end = week3_end.strftime('%Y-%m-%d')
        week4_end = datetime.strptime(week3_end, '%Y-%m-%d') + timedelta(days=7)
        week4_end = week4_end.strftime('%Y-%m-%d')

        week1 = actions[(actions['action_time']>=start_date) & (actions['action_time']<week1_end)] # 第一周的所有订单
        week1 = week1.drop_duplicates(['user_id'], keep='first')  # 第一周浏览了的所有用户
        week1 = week1[['user_id']]
        week1['week1'] = 1
        gc.collect()

        week2 = actions[(actions['action_time']>=week1_end) & (actions['action_time']<week2_end)] # 第二周的所有订单
        week2 = week2.drop_duplicates(['user_id'], keep='first')  # 第二周浏览了的所有用户
        week2 = week2[['user_id']]
        week2['week2'] = 1
        gc.collect()

        week3 = actions[(actions['action_time']>=week2_end) & (actions['action_time']<week3_end)] # 第二周的所有订单
        week3 = week3.drop_duplicates(['user_id'], keep='first')  # 第三周浏览了的所有用户
        week3 = week3[['user_id']]
        week3['week3'] = 1
        gc.collect()

        week4 = actions[(actions['action_time']>=week3_end) & (actions['action_time']<week4_end)] # 第二周的所有订单
        week4 = week4.drop_duplicates(['user_id'], keep='first')  # 第四周浏览了的所有用户
        week4 = week4[['user_id']]
        week4['week4'] = 1
        gc.collect()

        week = pd.merge(week1, week2, on=['user_id'], how='outer')
        week = pd.merge(week, week3, on=['user_id'], how='outer')
        week = pd.merge(week, week4, on=['user_id'], how='outer')
        del week1, week2, week3, week4
        gc.collect()
        # 用户连续浏览了几周
        week.fillna(0, inplace=True)
        week['week_read_num'] = week['week1']+week['week2']+week['week3']+week['week4']  # 用户浏览了几周
        week['week_read_lianxu'] = week['week4']
        week.loc[week['week_read_lianxu']==1, 'week_read_lianxu'] = week[week['week_read_lianxu']==1]['week_read_lianxu'] +\
                                                        week[week['week_read_lianxu']==1]['week3']
        week.loc[week['week_read_lianxu']==2, 'week_read_lianxu'] = week[week['week_read_lianxu']==2]['week_read_lianxu'] +\
                                                        week[week['week_read_lianxu']==2]['week2']                        
        week.loc[week['week_read_lianxu']==3, 'week_read_lianxu'] = week[week['week_read_lianxu']==3]['week_read_lianxu'] +\
                                                        week[week['week_read_lianxu']==3]['week1']
        week = week[['user_id', 'week_read_num', 'week_read_lianxu']]  # 用户连续浏览了几周
        
        df1 = pd.merge(df1, week, on='user_id', how='left')
        df1 = df1[['user_id', 'read_num', 'week_read_num', 'week_read_lianxu']]
#         pickle.dump(df1, open(dump_path, 'wb'))
    return df1


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   用户购买前访问天数  = 用户浏览天数 / 用户购买次数
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_user_feat6(start_date, end_date, actions_all):
    dump_path = './cache/user_feat6_{}_{}.pkl'.format(start_date, end_date)
    days = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
    days = int(days.days)  # 时间间隔的大小
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        # actions = actions_all.copy()
        actions = actions_all[['user_id', 'type', 'action_time']]
        actions['action_time'] = actions['action_time'].map(lambda x:x.split(' ')[0])
        visit = actions[actions['type']==1]
        visit = visit.drop_duplicates(['user_id', 'action_time'], keep='first')
        del visit['action_time'], actions['action_time']
        gc.collect()
        visit = visit.groupby('user_id', as_index=False).count()
        visit.columns = ['user_id', 'visit']

        buy = actions[actions['type'] == 2]
        buy = buy.groupby('user_id', as_index=False).count()
        buy.columns = ['user_id', 'buy']

        actions = pd.merge(visit, buy, on='user_id', how='left')
        actions['visit_day_before_buy'] = actions['visit'] / actions['buy']
        actions = actions[['user_id', 'visit_day_before_buy']]
#         pickle.dump(actions, open(dump_path, 'wb'))
    return actions


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   用户关注前访问天数  = 用户浏览天数 / 用户关注天数
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_user_feat7(start_date, end_date, actions_all):
    dump_path = './cache/user_feat7_{}_{}.pkl'.format(start_date, end_date)
    days = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
    days = int(days.days)  # 时间间隔的大小
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        # actions = actions_all.copy()
        actions = actions_all[['user_id', 'action_time', 'type']]
        actions['action_time'] = actions['action_time'].map(lambda x:x.split(' ')[0])
        visit = actions[actions['type']==1]
        visit = visit.drop_duplicates(['user_id', 'action_time'], keep='first')
        del visit['action_time'], actions['action_time']
        gc.collect()
        visit = visit.groupby('user_id', as_index=False).count()
        visit.columns = ['user_id', 'visit']

        guanzhu = actions[actions['type'] == 3]
        guanzhu = guanzhu.groupby('user_id', as_index=False).count()
        guanzhu.columns = ['user_id', 'guanzhu']

        actions = pd.merge(visit, guanzhu, on='user_id', how='left')
        actions['visit_day_before_guanzhu'] = actions['visit'] / actions['guanzhu']
        actions = actions[['user_id', 'visit_day_before_guanzhu']]
#         pickle.dump(actions, open(dump_path, 'wb'))
    return actions


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   用户购买前关注天数  = 用户关注天数 / 用户购买天数
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_user_feat8(start_date, end_date, actions_all):
    dump_path = './cache/user_feat8_{}_{}.pkl'.format(start_date, end_date)
    days = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
    days = int(days.days)  # 时间间隔的大小
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        # actions = actions_all.copy()
        actions = actions_all[['user_id', 'action_time', 'type']]
        actions['action_time'] = actions['action_time'].map(lambda x:x.split(' ')[0])
        guanzhu = actions[actions['type']==3]
        guanzhu = guanzhu.drop_duplicates(['user_id', 'action_time'], keep='first')
        del guanzhu['action_time'], actions['action_time']
        gc.collect()
        guanzhu = guanzhu.groupby('user_id', as_index=False).count()
        guanzhu.columns = ['user_id', 'guanzhu']

        buy = actions[actions['type'] == 2]
        buy = buy.groupby('user_id', as_index=False).count()
        buy.columns = ['user_id', 'buy']

        actions = pd.merge(guanzhu, buy, on='user_id', how='left')
        actions['guanzhu_day_before_buy'] = actions['guanzhu'] / actions['buy']
        actions = actions[['user_id', 'guanzhu_day_before_buy']]
#         pickle.dump(actions, open(dump_path, 'wb'))
    return actions

        
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   用户四种行为分别的 平均时间间隔
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_day_chaju(x, end_date):
    x = x.split(' ')[0]
    x = datetime.strptime(x, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    return (end_date - x).days

def get_user_feat9(start_date, end_date, actions_all):
    dump_path = './cache/user_feat9_{}_{}.pkl'.format(start_date, end_date)
    days = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
    days = int(days.days)  # 时间间隔的大小
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        # actions = actions_all.copy()
        actions = actions_all[actions_all['type']<5]
        actions = actions[['user_id', 'action_time', 'type']]
        actions['action_time'] = actions['action_time'].map(lambda x:(-1)*get_day_chaju(x, start_date))
        actions = actions.drop_duplicates(['user_id', 'action_time', 'type'], keep='first')
        actions = actions.groupby(['user_id', 'type']).agg(lambda x: np.diff(x).mean())
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
#         pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['user_id'] + ['user_feat9_'+str(i) for i in range(1, actions.shape[1])]
    return actions


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   用户四种行为的频率 = 行为的次数 / （最后一次时间-最早的时间）
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_user_feat10(start_date, end_date, actions_all):
    dump_path = './cache/user_feat10_{}_{}.pkl'.format(start_date, end_date)
    days = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
    days = int(days.days)  # 时间间隔的大小
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        # df = actions_all.copy()
        df = actions_all[actions_all['type']<5]
        df = df[['user_id', 'type', 'action_time']]
        df.rename(columns={'action_time': 'time'}, inplace=True)
        actions = df.groupby(['user_id', 'type'], as_index=False).count()

        time_min = df.groupby(['user_id', 'type'], as_index=False).min()
        time_max = df.groupby(['user_id', 'type'], as_index=False).max()

        time_cha = pd.merge(time_max, time_min, on=['user_id', 'type'], how='left')
        time_cha['time_x'] = pd.to_datetime(time_cha['time_x'])
        time_cha['time_y'] = pd.to_datetime(time_cha['time_y'])

        time_cha['cha_hour'] = 1 + (time_cha['time_x'] - time_cha['time_y']).dt.days * 24 + (time_cha['time_x'] -
                                                                                             time_cha[
                                                                                                 'time_y']).dt.seconds // 3600
        del time_cha['time_x']
        del time_cha['time_y']
        gc.collect()

        actions = pd.merge(time_cha, actions, on=['user_id', 'type'], how="left")
        actions = actions.groupby(['user_id', 'type']).sum()
        actions['cnt_divid_time'] = actions['time'] / actions["cha_hour"]
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        actions = actions.fillna(0)
#         pickle.dump(actions, open(dump_path, 'wb'))
    return actions


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   用户对品类的重复购买率
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_user_feat11(start_date, end_date, actions_all):
    dump_path = './cache/user_feat11_{}_{}.pkl'.format(start_date, end_date)
    days = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
    days = int(days.days)  # 时间间隔的大小
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        # df = actions_all.copy()
        df = actions_all[actions_all['type']==2]  # 所有购买行为
        # 对时间做一下去重
        df = df[['user_id', 'cate', 'action_time']]
        df['action_time'] = pd.to_datetime(df['action_time']).apply(lambda x:x.date())
        df = df.drop_duplicates(['user_id', 'cate', 'action_time'], keep='first')  # 保证用户对某个cate购买 每天最多一条记录
        df = df.groupby(['user_id', 'cate'], as_index=False).count()
        df.columns = ['user_id', 'cate', 'count1']
        df['count1'] = df['count1'].map(lambda x: 1 if x > 1 else 0)  # 用户对该cate是否重购过
        grouped = df.groupby(['user_id'], as_index=False)
        actions = grouped.count()[['user_id', 'count1']]
        actions.columns = ['user_id', 'count']
        re_count = grouped.sum()[['user_id', 'count1']]
        re_count.columns = ['user_id', 're_count']
        actions = pd.merge(actions, re_count, on='user_id', how='left')
        re_buy_rate = actions['re_count'] / actions['count']
        actions = pd.concat([actions['user_id'], re_buy_rate], axis=1)
        actions.columns = ['user_id', 're_buy_rate']
#         pickle.dump(actions, open(dump_path, 'wb'))
    return actions


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   用户最后一次行为的次数并且进行归一化
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_user_feat12(start_date, end_date, actions_all):
    dump_path = './cache/user_feat12_{}_{}.pkl'.format(start_date, end_date)
    days = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
    days = int(days.days)  # 时间间隔的大小
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        df = actions_all[['user_id', 'action_time', 'type']]
        df['action_time'] = df['action_time'].map(lambda x: get_day_chaju(x, end_date) + 1)

        idx = df.groupby(['user_id', 'type'])['action_time'].transform(min)
        idx1 = idx == df['action_time']
        actions = df[idx1].groupby(["user_id", "type"]).count()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.fillna(0)
        actions = actions.reset_index()

        user_sku = actions[['user_id']]
        del actions['user_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)
#         pickle.dump(actions, open(dump_path, 'wb'))
    return actions


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   根据module id 来计算用户各个行为的平均间隔天数
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# def get_user_feat12(start_date, end_date, actions_all):
#     dump_path = './cache/user_feat12_{}_{}.pkl'.format(start_date, end_date)
#     days = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
#     days = int(days.days)  # 时间间隔的大小
#     if os.path.exists(dump_path):
#         actions = pickle.load(open(dump_path, 'rb'))
#     else:
#         actions = actions_all[['user_id', 'action_time', 'module_id', 'type']]
#         actions = actions.drop_duplicates(['user_id', 'module_time'], keep='first')
