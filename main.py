from model import *
import pickle
train_start_date = '2018-03-01'  # 第一个训练集的结束时间
train_end_date = '2018-03-31'
train_set_path = './file/0613train_set_001.csv'

val_start_date = '2018-03-10'
val_end_date = '2018-04-09'
val_set_path = './file/0613val_set_001.csv'
val_label_path = './file/0613val_label_001.csv'

test_start_date = '2018-03-17'
test_end_date = '2018-04-16'
test_set_path = './file/0613test_set_001.csv'
submission_path = './file/submission_0613001.csv'

model_path = './file/xgb_0613_1.model'

xgb_param = {
    'eval_metric': 'auc',
    'max_depth': 3, 
    'eta': 0.0123, 
    'silent': 1, 
    'objective': 'binary:logistic',
    'learning_rate': 0.1,
    'seed': 2019,
    'subsample': 0.8,
    'colsample_bytree': 0.85,
    'nthread': -1,
    'scale_pos_weight': 25  # 设置正样本的权重
    # 'tree_method': 'gpu_hist',
    # 'device': 'gpu',  # 服务器不支持xgboost-gpu
}
# print('构造训练集...')
# make_n_train_set(train_start_date, 1, train_set_path)
# # make_train_set(train_start_date, train_end_date, train_set_path)
# # make_label(train_start_date, train_end_date, train_label_path)

print('构造验证集...')
# make_val_set(val_start_date, val_end_date, val_set_path)
# make_label(val_start_date, val_end_date, val_label_path)

print('构造测试集...')
# make_test_set(test_start_date, test_end_date, test_set_path)
# zero_cols = ['shop_cate', 'band_num', 'fans_num', 'vip_num', 'shop_score', 'have']
print('xgb模型训练...')
# # # zero_cols = pickle.load(open('./file/zero_cols.pkl', 'rb'))
# # # print("len(zero_cols): ", zero_cols)
# model, feats = xgb_train(xgb_param, train_set_path, model_path, zero_cols =[])
# # pickle.dump(feats, open('./file/zero_cols.pkl', 'wb'))
# # # zero_cols = pickle.load(open('./file/zero_cols.pkl', 'rb', 'rb')

# model = xgb.Booster(xgb_param) #init model
# model.load_model(model_path)
# # # show_feature_importance(model, feature_names)
models, features = xgb_train(xgb_param, train_set_path, val_set_path, val_label_path, model_path)

print('验证集效果...')
validate(model, val_set_path, val_label_path, verbose=False, zero_cols=[])

print('预测并生成预测结果...')
make_submission(models, test_set_path, submission_path, verbose=False, zero_cols=[])