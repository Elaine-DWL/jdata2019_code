# 1 cate-shop 行为i的记录数  / 该shop行为 i的记录数    [hc]
# 2 cate-shop 行为i的记录数  / 该cate行为 i的记录数    [hc]
# 4 当前shop里cate 的销量 占同类 shop_cate销量的比例


from common import *
import pickle
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   cate-shop 行为i的记录数  / 该shop行为 i的记录数
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


def get_cate_shop_feat1(start_date, end_date, actions_all):
    dump_path = './cache/cate_shop_feat1_{}_{}.pkl'.format(start_date, end_date)
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
            df1 = df.groupby(['cate', 'shop_id'], as_index=False)['type'].count()
            df = df.groupby(['shop_id'], as_index=False)['action_time'].count()
            df1 = pd.merge(df1, df, on=['shop_id'], how='outer')
            df1['cate_shop_action_{}_shop_ratio'.format(i)] = df1['type'] / df1['action_time']
            df1 = df1[['cate', 'shop_id', 'cate_shop_action_{}_shop_ratio'.format(i)]]
            if i == 1:
                _df = df1
            else:
                _df = pd.merge(_df, df1, on=['cate', 'shop_id'], how='outer')
        _df.fillna(0, inplace=True)
        pickle.dump(_df, open(dump_path, 'wb'))
    _df.columns = ['cate', 'shop_id'] + ['before_{}_{}'.format(days, col) for col in _df.columns if col not in ['cate', 'shop_id']]
    return _df


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   cate-shop 行为i的记录数  / 该cate行为 i的记录数
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_cate_shop_feat2(start_date, end_date, actions_all):
    dump_path = './cache/cate_shop_feat2_{}_{}.pkl'.format(start_date, end_date)
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
            df1 = df.groupby(['cate', 'shop_id'], as_index=False)['type'].count()
            df = df.groupby(['cate'], as_index=False)['action_time'].count()
            df1 = pd.merge(df1, df, on=['cate'], how='outer')
            df1['cate_shop_action_{}_cate_ratio'.format(i)] = df1['type'] / df1['action_time']
            df1 = df1[['cate', 'shop_id', 'cate_shop_action_{}_cate_ratio'.format(i)]]
            if i == 1:
                _df = df1
            else:
                _df = pd.merge(_df, df1, on=['cate', 'shop_id'], how='outer')
        _df.fillna(0, inplace=True)
        pickle.dump(_df, open(dump_path, 'wb'))
    _df.columns = ['cate', 'shop_id'] + ['before_{}_{}'.format(days, col) for col in _df.columns if col not in ['cate', 'shop_id']]
    return _df


