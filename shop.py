# 0 从shop表中构造出shop的基本属性
# 1 对shop的下单记录中 不同年龄用户记录数量的比重 & 该年龄段的下单记录中，购买该cate的记录比重
# 2 对shop的下单记录中 不同性别用户记录数量的比重 & 。。。。。。
# 3 对shop的下单记录中 不同城市等级用户记录数量的比重 & 。。。。。。
# 4 对shop的下单记录中 不同会员等级用户记录数量的比重 & 。。。。。。     [hc]
# 5 shop各个行为日均值 方差 转化率等
# 6 店铺累积评论属性
# 7 该shop 销量 占同类型店铺销量的比例   还没写



from common import *


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   shop基本属性
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_shop_feat0():
    dump_path = './cache/basic_shop.pkl'
    if os.path.exists(dump_path):
        shop = pickle.load(open(dump_path, 'rb'))
    else:
        shop = pd.read_csv(shop_path)
        shop['ziying'] = 0
        shop.loc[shop['vender_id'] == 3666, 'ziying'] = 1
        shop['cate'].fillna(-1, inplace=True)
        shop['shop_cate'] = shop['cate']
        del shop['cate']
        # shop['now'] = datetime.strptime('2018-02-01', '%Y-%m-%d')
        # shop['old'] = shop['shop_reg_tm'].apply(lambda x:datetime.strptime(x, '%Y-%m-%d'))
        # shop['old'] = shop['now'] - show['old']
        # shop['old'] = shop['old'].dt.days
        # shop['old'] = shop['old'].map(lambda x: int(x) // 365)
        # shop.rename(columns={'old': 'shop_year'}, inplace=True)
        # shop['vip_by_year'] = shop['vip_num'] // (shop['shop_year'] + 1)
        # shop['fans_by_year'] = shop['fans_num'] // (shop['shop_year'] + 1)
        # shop['pscore'] = shop['shop_score'] * np.log(1 + shop['shop_year'])
        # shop = shop[['shop_id', 'fans_num', 'vip_num', 'ziying', 'pscore', 'vip_by_year', 'fans_by_year', 'shop_score',
        #              'shop_cate']]
        shop = shop[['shop_id', 'fans_num', 'vip_num', 'ziying','shop_score', 'shop_cate']]
        gc.collect()
        pickle.dump(shop, open(dump_path, 'wb'))
    return shop


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   对shop下单记录中 不同年龄用户比例
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_shop_feat1(start_date, end_date, actions_all):
    dump_path = './cache/shop_feat1_{}_{}.pkl'.format(start_date, end_date)
    if os.path.exists(dump_path):
        _df = pickle.load(open(dump_path, 'rb'))
    else:
        # actions = get_actions(start_date, end_date)
        actions = actions_all.copy()
        actions = actions[actions['type'] == 2]
        gc.collect()
        user = pd.read_csv(user_path)
        user['age'].fillna(-1.0, inplace=True)
        actions = pd.merge(actions, user[['user_id', 'age']], on='user_id', how='left')
        df = actions.groupby(['shop_id', 'age'], as_index=False)['type'].count()  # 品类-属性人群记录数量
        df1 = actions.groupby(['shop_id'], as_index=False)['action_time'].count()  # 品类记录数量
        df2 = actions.groupby(['age'], as_index=False)['sku_id'].count()
        _df = pd.merge(df, df1, on='shop_id', how='left')
        _df = pd.merge(_df, df2, on='age', how='left')

        _df['shop_age_ratio'] = _df['type'] / _df['action_time']
        _df['age_shop_ratio'] = _df['type'] / _df['sku_id']
        _df = _df[['shop_id', 'age', 'shop_age_ratio', 'age_shop_ratio']]
        _df.fillna(0, inplace=True)
        pickle.dump(_df, open(dump_path, 'wb'))
    return _df


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   对shop下单记录中 不同性别用户比例
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_shop_feat2(start_date, end_date, actions_all):
    dump_path = './cache/shop_feat2_{}_{}.pkl'.format(start_date, end_date)
    if os.path.exists(dump_path):
        _df = pickle.load(open(dump_path, 'rb'))
    else:
        # actions = get_actions(start_date, end_date)
        actions = actions_all.copy()
        actions = actions[actions['type'] == 2]
        gc.collect()
        user = pd.read_csv(user_path)
        user['sex'].fillna(-1.0, inplace=True)
        actions = pd.merge(actions, user[['user_id', 'sex']], on='user_id', how='left')
        df = actions.groupby(['shop_id', 'sex'], as_index=False)['type'].count()  # 品类-属性人群记录数量
        df1 = actions.groupby(['shop_id'], as_index=False)['action_time'].count()  # 品类记录数量
        df2 = actions.groupby(['sex'], as_index=False)['sku_id'].count()
        _df = pd.merge(df, df1, on='shop_id', how='left')
        _df = pd.merge(_df, df2, on='sex', how='left')

        _df['shop_sex_ratio'] = _df['type'] / _df['action_time']
        _df['sex_shop_ratio'] = _df['type'] / _df['sku_id']
        _df = _df[['shop_id', 'sex', 'shop_sex_ratio', 'sex_shop_ratio']]
        _df.fillna(0, inplace=True)
        pickle.dump(_df, open(dump_path, 'wb'))
    return _df


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   对shop下单记录中 不同城市等级用户比例
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_shop_feat3(start_date, end_date, actions_all):
    dump_path = './cache/shop_feat3_{}_{}.pkl'.format(start_date, end_date)
    if os.path.exists(dump_path):
        _df = pickle.load(open(dump_path, 'rb'))
    else:
        # actions = get_actions(start_date, end_date)
        actions = actions_all.copy()
        actions = actions[actions['type'] == 2]
        gc.collect()
        user = pd.read_csv(user_path)
        user['city_level'].fillna(-1.0, inplace=True)
        actions = pd.merge(actions, user[['user_id', 'city_level']], on='user_id', how='left')
        df = actions.groupby(['shop_id', 'city_level'], as_index=False)['type'].count()  # 品类-属性人群记录数量
        df1 = actions.groupby(['shop_id'], as_index=False)['action_time'].count()  # 品类记录数量
        df2 = actions.groupby(['city_level'], as_index=False)['sku_id'].count()
        _df = pd.merge(df, df1, on='shop_id', how='left')
        _df = pd.merge(_df, df2, on='city_level', how='left')

        _df['shop_city_level_ratio'] = _df['type'] / _df['action_time']
        _df['city_level_shop_ratio'] = _df['type'] / _df['sku_id']
        _df = _df[['shop_id', 'city_level', 'shop_city_level_ratio', 'city_level_shop_ratio']]
        _df.fillna(0, inplace=True)
        pickle.dump(_df, open(dump_path, 'wb'))
    return _df


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   对shop下单记录中 不同会员等级用户比例
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_shop_feat4(start_date, end_date, actions_all):
    dump_path = './cache/shop_feat4_{}_{}.pkl'.format(start_date, end_date)
    if os.path.exists(dump_path):
        _df = pickle.load(open(dump_path, 'rb'))
    else:
        # actions = get_actions(start_date, end_date)
        actions = actions_all.copy()
        actions = actions[actions['type'] == 2]
        gc.collect()
        user = pd.read_csv(user_path)
        user['user_lv_cd'].fillna(-1.0, inplace=True)
        actions = pd.merge(actions, user[['user_id', 'user_lv_cd']], on='user_id', how='left')
        df = actions.groupby(['shop_id', 'user_lv_cd'], as_index=False)['type'].count()  # 品类-属性人群记录数量
        df1 = actions.groupby(['shop_id'], as_index=False)['action_time'].count()  # 品类记录数量
        df2 = actions.groupby(['user_lv_cd'], as_index=False)['sku_id'].count()
        _df = pd.merge(df, df1, on='shop_id', how='left')
        _df = pd.merge(_df, df2, on='user_lv_cd', how='left')

        _df['shop_user_lv_cd_ratio'] = _df['type'] / _df['action_time']
        _df['user_lv_cd_shop_ratio'] = _df['type'] / _df['sku_id']
        _df = _df[['shop_id', 'user_lv_cd', 'shop_user_lv_cd_ratio', 'user_lv_cd_shop_ratio']]
        _df.fillna(0, inplace=True)
        pickle.dump(_df, open(dump_path, 'wb'))
    return _df


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   shop各个行为日均值 方差 转化率等
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_shop_feat5(start_date, end_date, actions_all):
    dump_path = './cache/shop_feat5_{}_{}.pkl'.format(start_date, end_date)
    days = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
    days = int(days.days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        features = ['shop_id', 'shop_action_1_ratio', 'shop_action_3_ratio', 'shop_action_4_ratio']
        
        # actions = get_actions(start_date, end_date)
        actions = actions_all[['shop_id', 'type']]
        actions = actions[actions['type']<5]
        df = pd.get_dummies(actions['type'], prefix='shop_action')
        cols = ['shop_action_{}'.format(i) for i in range(1, 5)]
        for col in cols:
            if col not in df.columns:
                df[col] = 0
        actions = pd.concat([actions[['shop_id']], df], axis=1)
        actions = actions.groupby(['shop_id'], as_index=False).sum()
        actions['shop_action_1_ratio'] = (np.log(1 + actions['shop_action_2']) - np.log(1 + actions['shop_action_1']))
        actions['shop_action_3_ratio'] = (np.log(1 + actions['shop_action_2']) - np.log(1 + actions['shop_action_3']))
        actions['shop_action_4_ratio'] = (np.log(1 + actions['shop_action_2']) - np.log(1 + actions['shop_action_4']))
    
        actions = actions[features]
        pickle.dump(actions, open(dump_path, 'wb'))
    actions.columns = ['shop_id'] + ['before_{}_{}'.format(days, col) for col in actions.columns if col not in ['shop_id']]
    return actions


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                   该shop 到最后日期的评论数
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_shop_feat6(start_date, end_date, actions_all):
    dump_path = './cache/shop_feat6_{}_{}.pkl'.format(start_date, end_date)
    if os.path.exists(dump_path):
        comment = pickle.load(open(dump_path, 'rb'))
    else:
        comment = pd.read_csv(comment_path)
        product = pd.read_csv(product_path)
        comment = pd.merge(comment, product[['shop_id', 'sku_id']], on='sku_id', how='left')
        # 统计截止到某一天 评论总数
        comment = comment.groupby(['shop_id', 'dt'], as_index=False)[
                'comments', 'good_comments', 'bad_comments'].sum()
        comment = comment.sort_values(by=['shop_id', 'dt'])
        cumsum = comment.groupby('shop_id')['comments', 'good_comments', 'bad_comments'].cumsum()
        cumsum.columns = ["{}_cumsum".format(col) for col in cumsum.columns]
        comment = pd.merge(comment, cumsum, left_index=True, right_index=True)
        comment.fillna(0, inplace=True)
        del cumsum
        gc.collect()
        comment['have'] = 0;
        comment.loc[comment['dt'] == end_date, 'have'] = 1
        comment.loc[comment['have'] == 1, 'comments_cumsum'] = comment.loc[comment['have'] == 1, 'comments_cumsum'] - \
                                                               comment.loc[comment['have'] == 1, 'comments']
        comment.loc[comment['have'] == 1, 'good_comments_cumsum'] = comment.loc[comment['have'] == 1, 'good_comments_cumsum'] - \
                                                                    comment.loc[comment['have'] == 1, 'good_comments']
        comment.loc[comment['have'] == 1, 'bad_comments_cumsum'] = comment.loc[comment['have'] == 1, 'bad_comments_cumsum'] - \
                                                                   comment.loc[comment['have'] == 1, 'bad_comments']
        comment = comment[comment.dt <= end_date]
        comment = comment.sort_values(by=['shop_id', 'dt'])
        comment = comment.groupby(['shop_id'], as_index=False).last()
        gc.collect()
        # 计算差评率
        comment['bad_rate'] = 0
        comment['bad_rate'] = comment['bad_comments_cumsum'] / comment['comments_cumsum']
        comment['bad_rate'].fillna(0.0, inplace=True)
        comment.replace(np.inf, 0.0, inplace=True)
        def judge(x):
            if x == 0:
                    return 0
            else:
                    return 1

        comment['have_bad'] = comment['bad_comments_cumsum'].apply(judge)
        comment.drop(['dt', 'comments', 'good_comments', 'bad_comments'], axis=1, inplace=True)
        pickle.dump(comment, open(dump_path, 'wb'))
    return comment