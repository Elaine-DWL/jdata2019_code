# 1 用户对该cate的行为i记录数 / 该用户的所有下单行为记录数   （4种类行为）    [hc]
# 2 用户对该cate 有行为i的天数/ 时间区间间隔天数  （4种行为）                [hc]
# 3 用户对该cate行为i的最后时间  距离end_date有多少天
# 4 用户最后对该cate 产生行为的最后时间 距离end_date有多少天


from common import *


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   用户对该cate购买行为数x  / 用户的所有购买行为数y   x可以是x1/x2/x3/x4
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_user_cate_feat1(start_date, end_date, actions_all):
    # 时间区间内用户对该cate的购买次数
    dump_path = './cache/user_cate_feat1_{}_{}.pkl'.format(start_date, end_date)
    days = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
    days = int(days.days)
    if os.path.exists(dump_path):
        _df = pickle.load(open(dump_path, 'rb'))
    else:
        # actions = get_actions(start_date, end_date)
        actions = actions_all.copy()
        _df = None
        for i in range(1, 5):
            df = actions[actions['type'] == i]
            df = df.groupby(['user_id', 'cate'], as_index=False)['type'].count()  # 用户对该cate的行为i记录数目
            df.rename(columns={'type': 'user_cate_action_{}_cnt'.format(i)}, inplace=True)
            df1 = actions.groupby(['user_id'], as_index=False)['cate'].count()  # 用户总的行为i记录数目
            df1.rename(columns={'cate': 'user_action_{}_cnt'.format(i)}, inplace=True)
            df = pd.merge(df, df1, on='user_id', how='right')
            df['user_cate_action_{}_rate'.format(i)] = df['user_cate_action_{}_cnt'.format(i)] / df['user_action_{}_cnt'.format(i)]
            df = df[['user_id', 'cate', 'user_cate_action_{}_rate'.format(i)]]
            if i == 1:
                _df = df
            else:
                _df = pd.merge(_df, df, on=['user_id', 'cate'], how='outer')
        _df.fillna(0, inplace=True)
        pickle.dump(_df, open(dump_path, 'wb'))
    _df.columns = ['user_id', 'cate'] + ['before_{}_{}'.format(days, col) for col in _df.columns if col not in ['user_id', 'cate']]
    return _df


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   用户对该cate的平均访问/下单/评论/关注时间间隔【单位：天】
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def get_user_cate_feat2(start_date, end_date, actions_all):
    dump_path = './cache/user_cate_feat2_{}_{}.pkl'.format(start_date, end_date)
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
            df = df[['user_id', 'cate', 'action_time']]
            df['action_time'] = pd.to_datetime(df['action_time']).apply(lambda x:x.date())
            df = df.drop_duplicates(['user_id', 'cate', 'action_time'], keep='first')
            df = df.sort_values(by=['user_id', 'cate', 'action_time'])
            df = df.groupby(['user_id', 'cate'], as_index=False)['action_time'].count()  # 用户对该cate 行为i的天数
            df['user_cate_avg_action_{}_day'.format(i)] = df['action_time'] / days
            df = df[['user_id', 'cate', 'user_cate_avg_action_{}_day'.format(i)]]
            if i == 1:
                _df = df
            else:
                _df = pd.merge(_df, df, on=['user_id', 'cate'], how='outer')
        _df.fillna(days, inplace=True)
        pickle.dump(_df, open(dump_path, 'wb'))
    _df.columns = ['user_id', 'cate'] + ['before_{}_{}'.format(days, col) for col in _df.columns if col not in ['user_id', 'cate']]
    return  _df


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   用户最后对该cate 产生行为i的时间距离 end_date有多少天
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_user_cate_feat3(start_date, end_date, actions_all):
    dump_path = './cache/user_cate_feat3_{}_{}.pkl'.format(start_date, end_date)
    if os.path.exists(dump_path):
        _df = pickle.load(open(dump_path, 'rb'))
    else:
        # actions = get_actions(start_date, end_date)
        actions = actions_all.copy()
        actions['action_time'] = pd.to_datetime(actions['action_time']).apply(lambda x:x.date())
        _df = None
        for i in range(1, 5):
            df = actions[actions['type'] == i]
            df = df.sort_values(by=['user_id', 'action_time'])
            df = df.groupby(['user_id', 'cate'], as_index=False)['action_time'].last()
            df['end_date'] = end_date
            df['end_date'] = pd.to_datetime(df['end_date']).apply(lambda x:x.date())
            df['user_cate_action_{}_last_diff'.format(i)] = df['end_date'] - df['action_time']
            df['user_cate_action_{}_last_diff'.format(i)] = df['user_cate_action_{}_last_diff'.format(i)].map(lambda x:int(x.days))
            df = df[['user_id', 'cate', 'user_cate_action_{}_last_diff'.format(i)]]
            gc.collect()
            if i == 1:
                _df = df
            else:
                _df = pd.merge(_df, df, on=['user_id', 'cate'], how='outer')
        _df.fillna(30, inplace=True)
        pickle.dump(_df, open(dump_path, 'wb'))
    # _df.columns = ['user_id'] + ['u_s_feat3_'+ str(i) for i in range(1, _df.shape[1])]
    return _df


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   用户最后对该cate 产生行为时间距离 end_date有多少天
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_user_cate_feat4(start_date, end_date, actions_all):
    dump_path = './cache/user_cate_feat4_{}_{}.pkl'.format(start_date, end_date)
    if os.path.exists(dump_path):
        df = pickle.load(open(dump_path, 'rb'))
    else:
        # actions = get_actions(start_date, end_date)
        actions = actions_all.copy()
        actions['action_time'] = pd.to_datetime(actions['action_time']).apply(lambda x:x.date())
        df = actions.sort_values(by=['user_id', 'action_time'])
        df = df.groupby(['user_id', 'cate'], as_index=False)['action_time'].last()
        df['end_date'] = end_date
        df['end_date'] = pd.to_datetime(df['end_date']).apply(lambda x:x.date())
        df['user_cate_action_last_diff'] = df['end_date'] - df['action_time']
        df['user_cate_action_last_diff'] = df['user_cate_action_last_diff'].map(lambda x:int(x.days))
        df = df[['user_id', 'cate', 'user_cate_action_last_diff']]
        pickle.dump(df, open(dump_path, 'wb'))
    # _df.columns = ['user_id'] + ['u_s_feat3_'+ str(i) for i in range(1, _df.shape[1])]
    return df