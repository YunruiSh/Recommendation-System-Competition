# DSCI553-Competition
# Yunrui Shao
# Student_id: 8706491261

# import
import pyspark
from pyspark import SparkContext
import sys, json, csv, time, datetime, math, random, binascii
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
warnings.filterwarnings(action='ignore')

# user inputs
folder_path = sys.argv[1]
test_fpath = sys.argv[2]
output_fpath = sys.argv[3]
sc = SparkContext('local[*]', 'task')
sc.setLogLevel('ERROR')

# start time
start_time = time.time()

# load RDD
train_RDD = sc.textFile(folder_path+'yelp_train.csv', 30)
header = train_RDD.first()
train_RDD = train_RDD.filter(lambda x:x!=header)
test_RDD = sc.textFile(test_fpath, 30)
header = test_RDD.first()
test_RDD = test_RDD.filter(lambda x:x!=header)

# create user and business
# helper function
def map_user(sub, bol): 
    return sub.map(lambda x:(x.split(',')[bol], 1)).reduceByKey(lambda x,y:x).map(lambda x:(1, [x[0]])).reduceByKey(lambda x,y:x+y).collect()[0][1]
# continue
user_dict = {}
bus_dict = {}
user = list(set(map_user(train_RDD, 0) + map_user(test_RDD, 0)))
for x in user: user_dict[x]={}
bus = list(set(map_user(train_RDD, 1) + map_user(test_RDD, 1)))
for x in bus: bus_dict[x]={}
# deal with bus
bus_RDD = sc.textFile(folder_path+'business.json', 30)
bus_tlist = ['business_id', 'latitude', 'longitude', 'stars', 'review_count', 'is_open', 'attributes', 'categories', 'hours', 'state']
# helper
def deNone(x, bol):
    result = 0
    if x!=None and x!="None":
        if bol==0: result = len(x)
        if bol==1: result = len(x.split(','))
        if bol==2: result = int(sorted(x.split(','), reverse=True)[0])
    return result
# continue
bus_save = bus_RDD.map(lambda x:json.loads(x)).map(lambda y:(y[bus_tlist[0]], 
                                                             y[bus_tlist[1]], 
                                                             y[bus_tlist[2]],
                                                             y[bus_tlist[3]],
                                                             y[bus_tlist[4]],
                                                             y[bus_tlist[5]],
                                                             deNone(y[bus_tlist[6]],0),
                                                             deNone(y[bus_tlist[7]],1),
                                                             deNone(y[bus_tlist[8]],0),
                                                             y[bus_tlist[9]])).collect()
tmp = [9, 1, 2, 3, 4, 5, 6, 7, 8]
bus_tlist2 = ['business_id', 'latitude', 'longitude', 'stars', 'review_count', 'is_open', 'len_attributes', 'len_categories', 'len_hours', 'state']
for x in bus_save:
    try:
        for y in tmp:
             bus_dict[x[0]][bus_tlist2[y]] = x[y]
    except:
        continue
bus_RDD = sc.textFile(folder_path+'business.json', 30)
bus_save = bus_RDD.map(lambda x:json.loads(x)).map(lambda y:(y[bus_tlist[0]], y[bus_tlist[7]])).collect()    
cat = "category"
bus_pd = pd.DataFrame(bus_save, columns=[bus_tlist[0], cat])
bus_pd[cat] = bus_pd[cat].fillna('').apply(lambda x:x.split(', '))
save1 = []
for x in bus_pd[cat]: save1+=x
save2 = set(save1)
# helper function
def inOrNot(x):
    if s in x: return 1
    else: return 0
# continue
for s in save2: bus_pd[s] = bus_pd[cat].apply(inOrNot)
# helper function
def helpDict(val):
    result = dict()
    if val == None: 
        return {}
    for k, v in val.items():
        if "{" in v:
            tmp = [y.split(": ") for y in v.replace("'", "").replace("{", "").replace("}", "").split(", ")]
            for x in tmp:
                if x[1].isdigit() == True: result[x[0]] = int(x[1])
                else:result[x[0]] = x[1]
        else:
            if v.isdigit() == True: result[k] = int(v)
            else: result[k] = v
    return result
# continue
bus_RDD = sc.textFile(folder_path+'business.json', 30)
bus_save = bus_RDD.map(lambda x:json.loads(x)).map(lambda y:(y['business_id'], y['attributes'])).map(lambda z:(z[0], helpDict(z[1]))).collect()
save3 = dict()
for a, b in bus_save: save3[a] = b
a_pd = pd.DataFrame(save3).T
a_pd = pd.get_dummies(a_pd, drop_first=True)
# PCA
pca = PCA(n_components=5).fit_transform(a_pd)
a_pd = pd.DataFrame(pca, index=a_pd.index, columns=['attr_pca_' + str(x+1) for x in range(5)])
bus_save = pd.DataFrame(PCA(n_components=10).fit_transform(bus_pd.iloc[:, 3:]), columns=['pca_' + str(x+1) for x in range(10)])
bus_save['business_id'] = bus_pd['business_id']
for a, b in bus_save.iterrows():
    try:
        for x in range(10):
            bus_dict[b['business_id']]['category_pca_'+str(x+1)] = b['pca_'+str(x+1)]
    except:
        continue
# user
user_tlist = ['review_count', 'date_since', 'n_friends', 'useful', 'funny', 'fans', 'n_elite', 'max_elite', 'avg_stars', 'compliment_hot', 'compliment_more', 'compliment_cute', 'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool', 'compliment_funny', 'compliment_writer', 'compliment_photos']    
user_RDD = sc.textFile(folder_path + 'user.json', 30).map(lambda x:json.loads(x)).filter(lambda y:y['user_id'] in user)
tmp = 'yelping_since'
user_save = user_RDD.map(lambda x:(x['user_id'], x[user_tlist[0]], (datetime.date(2021, 3, 10) - datetime.date(int(x[tmp].split('-')[0]), int(x[tmp].split('-')[1]), int(x[tmp].split('-')[2]))).days, \
                                   deNone(x['friends'], 1), x[user_tlist[3]], x[user_tlist[4]], x[user_tlist[5]], \
                                   deNone(x['elite'], 1), deNone(x['elite'], 2), x['average_stars'], \
                                   x[user_tlist[9]], x[user_tlist[10]], x[user_tlist[11]], x[user_tlist[12]], x[user_tlist[13]], x[user_tlist[14]], x[user_tlist[15]], x[user_tlist[16]], x[user_tlist[17]], x[user_tlist[18]])).collect()
for x in user_save:
    try:
        for y in range(1,20):
             user_dict[x[0]][user_tlist[y-1]] = x[y]
    except:
        continue
# checkin
check_RDD = sc.textFile(folder_path + 'checkin.json', 30)
# helper function
def helpOperator(x, bol):
    result = 0
    for y in x: 
        result+=y
    result1 = result
    if bol==1: result1 = result/len(x)
    return result1
# continue
check_save = check_RDD.map(lambda x:json.loads(x)).map(lambda x:(x['business_id'], x['time'])).map(lambda x:(x[0], list(x[1].values()))).map(lambda x:(x[0], helpOperator(x[1], 0), helpOperator(x[1], 1))).collect()
save4 = ['checkin_sum', 'checkin_avg']
for x in check_save:
    try:
        bus_dict[x[0]]["checkin_sum"] = x[1]
        bus_dict[x[0]]["checkin_avg"] = x[2]
    except:
        continue
# tip
tip_RDD = sc.textFile(folder_path + 'tip.json', 30)
# helper function
def tipS(word):
    result = tip_RDD.map(lambda x:json.loads(x)).map(lambda x:(x[word], (1, x['likes'], len(x['text']))))
    result = result.reduceByKey(lambda a, b:(a[0]+b[0], a[1]+b[1], a[2]+b[2])).map(lambda x:(x[0], x[1][0], x[1][1]/x[1][0], x[1][2]/x[1][0])).collect()
    return result
tip_save = tipS('business_id')
tip_list = ['0', 'n_tip_business', 'avg_like_business', 'avg_tip_len_business']
for x in tip_save:
    try:
        for y in range(1,4):
            bus_dict[x[0]][tip_list[y]] = x[y]
    except:
        pass
tip_save = tipS('user_id')
tip_list = ['0', 'n_tip_user', 'avg_like_user', 'avg_tip_len_user']
for x in tip_save:
    try:
        for y in range(1,4):
            user_dict[x[0]][tip_list[y]] = x[y]
    except:
        pass
# train
save5 = []
tmp = 0
train_save = train_RDD.map(lambda x:(x.split(',')[0], x.split(',')[1], x.split(',')[2])).map(lambda x:(x[0], x[1], user_dict[x[0]], bus_dict[x[1]], a_pd.loc[x[1]], float(x[2]))).collect()
for x in train_save:
    save6 = {}
    for i in x[2].items(): save6[i[0]] = i[1]
    for i in x[3].items(): save6[i[0]] = i[1]
    for i, j in enumerate(x[4]): save6['bus_attr_' + str(i)] = j
    save6['target'] = x[5]
    save5.append(save6)
    tmp += 1
train_pd = pd.DataFrame.from_dict(save5)
# test
save7 = []
tmp = 0
test_save = test_RDD.map(lambda x:(x.split(',')[0], x.split(',')[1])).map(lambda x:(x[0], x[1], user_dict[x[0]], bus_dict[x[1]], a_pd.loc[x[1]])).collect()
for x in test_save:
    save8 = {}
    for i in x[2].items(): save8[i[0]] = i[1]
    for i in x[3].items(): save8[i[0]] = i[1]
    for i, j in enumerate(x[4]): save8['bus_attr_' + str(i)] = j
    save7.append(save8)
    tmp += 1
test_pd = pd.DataFrame.from_dict(save7)
# continue
result = []
trainy = train_pd['target']
trainx = train_pd.loc[:, train_pd.columns != 'target']
testx = test_pd.iloc[:, :]
data = pd.get_dummies(pd.concat(objs=[trainx, testx], axis=0), drop_first=True)
trainx = data[:len(trainx)][:]
testx = data[len(trainx):][:]
xgbst = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.7, learning_rate=0.07, max_depth=8, n_estimators=300, subsample=1.0, random_state=0, min_child_weight=4, reg_alpha=0.5, reg_lambda=0.5)
xgbst.fit(trainx, trainy, eval_metric='rmse', eval_set=[(trainx, trainy)], early_stopping_rounds=5)
pred = xgbst.predict(testx)
for x, y in zip(test_save, pred):
    if y<1: result.append((x[0], x[1], 1.0))
    elif y>5: result.append((x[0], x[1], 5.0))
    else: result.append((x[0], x[1], y))

# end time
runtime = time.time() - start_time
print('Duration: ', runtime)

# print
with open(output_fpath, 'w') as f:
    pointer = csv.writer(f, delimiter=',')
    pointer.writerow(['user_id', 'business_id', 'prediction'])
    for l in result:
        pointer.writerow([l[0], l[1], l[2]])

# spark-submit competition.py <folder_path> <test_file_name> <output_file_name>
# spark-submit competition.py "file:/Users/shaoyunrui/Desktop/553-comp/CompetitionStudentData/" "yelp_val.csv" "output.csv"