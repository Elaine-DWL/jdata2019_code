# 1 用户对shop的行为i记录数 / 用户行为总数          [hc]
# 2 用户对shop 的行为i的天数 / 时间段天数           [hc]
# 3 用户最后对该shop 产生行为i的时间距离 end_date有多少天


from common import *


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   用户对shop的行为i记录数 /  用户行为i总数    
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_user_shop_feat1(start_date, end_date, actions_all):
    dump_path = './cache/user_shop_feat1_{}_{}.pkl'.format(start_date, end_date)
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
            df = df.groupby(['user_id', 'shop_id'], as_index=False)['type'].count()  # 用户在该店铺下单数量
            df.rename(columns={'type': 'user_shop_action_{}_cnt'.format(i)}, inplace=True)
            df1 = actions.groupby(['user_id'], as_index=False)['cate'].count()  # 用户总的下单数量
            df1.rename(columns={'cate': 'user_action_{}_cnt'.format(i)}, inplace=True)
            df = pd.merge(df, df1, on='user_id', how='left')
            df['user_shop_action_{}_rate'.format(i)] = df['user_shop_action_{}_cnt'.format(i)] / df['user_action_{}_cnt'.format(i)]
            df = df[['user_id', 'shop_id', 'user_shop_action_{}_rate'.format(i)]]
            if i==1:
                _df = df
            else:
                _df = pd.merge(_df, df, on=['user_id', 'shop_id'], how='outer')
        _df.fillna(0, inplace=True)
        pickle.dump(_df, open(dump_path, 'wb'))
    _df.columns = ['user_id', 'shop_id'] + ['before_{}_{}'.format(days, col) for col in _df.columns if col not in ['user_id', 'shop_id']]
    return _df


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   用户在该店的行为i的天数x /  时间段天数    
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_user_shop_feat2(start_date, end_date, actions_all):
    dump_path = './cache/user_shop_feat2_{}_{}.pkl'.format(start_date, end_date)
    days = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
    days = int(days.days)
    if os.path.exists(dump_path):
            _df = pickle.load(open(dump_path, 'rb'))
    else:
        # actions = get_actions(start_date, end_date)
        actions = actions_all.copy()
        actions['action_time'] = pd.to_datetime(actions['action_time']).apply(lambda x: x.date())
        _df = None
        for i in range(1, 5):
            df = actions[actions['type'] == i]
            df = df.groupby(['user_id', 'shop_id'])['action_time'].nunique().reset_index()  # 用户对该店行为i的天数
            df['user_shop_avg_action_{}_day'.format(i)] =  df['action_time'] / days
            df = df[['user_id', 'shop_id', 'user_shop_avg_action_{}_day'.format(i)]]
            if i == 1:
                _df = df
            else:
                _df = pd.merge(_df, df, on = ['user_id', 'shop_id'], how='outer')
        _df.fillna(0, inplace=True)
        pickle.dump(_df, open(dump_path, 'wb'))
    _df.columns = ['user_id', 'shop_id'] + ['before_{}_{}'.format(days, col) for col in _df.columns if col not in ['user_id', 'shop_id']]
    return _df


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   用户最后对该shop 产生行为i的时间距离 end_date有多少天
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_user_shop_feat3(start_date, end_date, actions_all):
    dump_path = './cache/user_shop_feat3_{}_{}.pkl'.format(start_date, end_date)
    if os.path.exists(dump_path):
            _df = pickle.load(open(dump_path, 'rb'))
    else:
        # actions = get_actions(start_date, end_date)
        actions = actions_all.copy()
        actions['action_time'] = pd.to_datetime(actions['action_time']).apply(lambda x: x.date())
        _df = None
        for i in range(1, 5):
            df = actions[actions['type'] == i]
            df = df.sort_values(by=['user_id', 'action_time'])
            df = df.groupby(['user_id', 'shop_id'], as_index=False)['action_time'].last()
            df['end_date'] = end_date
            df['end_date'] = pd.to_datetime(df['end_date']).apply(lambda x:x.date())
            df['user_shop_action_{}_last_diff'.format(i)] = df['end_date'] - df['action_time']
            df['user_shop_action_{}_last_diff'.format(i)] = df['user_shop_action_{}_last_diff'.format(i)].map(lambda x:int(x.days))
            df = df[['user_id', 'shop_id', 'user_shop_action_{}_last_diff'.format(i)]]
            gc.collect()
            if i == 1:
                _df = df
            else:
                _df = pd.merge(_df, df, on=['user_id', 'shop_id'], how='outer')
        _df.fillna(30, inplace=True)
        pickle.dump(_df, open(dump_path, 'wb'))
    # _df.columns = ['user_id'] + ['u_s_feat3_'+ str(i) for i in range(1, _df.shape[1])]
    return _df

