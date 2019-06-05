#!/usr/bin/env python
# encoding: utf-8
# 数据预处理
# 清洗 并完成两步网格划分

import time
import pandas as pd
import numpy as np
from copy import deepcopy
import math
import multiprocessing
import decimal
from joblib import Parallel, delayed

dict = []
# 州范围
state_list = [[49.105479, -124.867437, 46.025512, -117.012127], [46.111500, -124.557297, 42.170991, -117.115407],
              [48.950396, -117.019754, 42.030739, -104.161416], [41.939908, -113.990605, 31.598750, -109.225379],
              [42.047690, -108.976738, 31.640440, -103.254583], [42.023995, -124.766806, 34.676570, -120.133517],
              [41.939860, -119.874728, 37.460597, -114.108092], [36.626321, -121.856734, 32.603269, -114.996902],
              [48.999698, -103.991301, 40.988367, -96.699520], [36.497755, -94.648962, 33.131238, -91.234274],
              [36.414605, -89.587018, 35.130495, -84.446850], [33.024059, -94.366532, 29.746666, -91.320693],
              [35.019828, -90.998006, 30.148443, -88.398885], [34.995270, -88.140430, 31.100974, -85.187000],
              [34.949453, -85.554362, 30.621492, -82.084702], [30.540419, -83.194550, 25.276958, -80.313269],
              [36.577615, -83.053887, 34.858894, -77.039700], [35.177500, -81.938587, 32.329808, -78.674591],
              [49.017011, -97.119000, 43.626560, -91.308115], [43.449056, -96.463415, 40.661157, -91.427940],
              [40.536053, -95.699495, 36.635581, -90.184865], [42.506427, -91.509767, 36.777139, -87.666879],
              [41.598007, -87.506397, 36.679703, -81.088372], [41.960883, -80.457877, 39.787819, -75.828910],
              [45.044420, -73.302403, 41.385284, -70.819510], [47.236566, -70.458025, 43.274077, -67.355996],
              [42.013585, -75.316450, 39.791879, -73.733551], [45.092751, -79.052341, 42.069116, -73.651948],
              [48.074244, -92.304710, 42.076842, -83.264707], [36.840414, -102.854631, 26.282009, -94.100531]]
# 得克萨斯州的数据大概为90万，城市分别为奥斯汀 329453,7828,、圣安东尼奥 31130,2035、休斯顿 65763,3351、达拉斯 116787,4664
# city_list = [[30.505091, -97.938174, 30.142959, -97.569989], [29.741792, -98.793220, 29.236319, -98.328551],
#              [30.105264, -95.753117, 29.541228, -95.075643], [33.002712, -97.006228, 32.627882, -96.560694]]

# city_list = [[30.326427, -97.775325, 30.296427, -97.744585]]
# city_list = [[29.816691, -95.456244, 29.679229, -95.286390]]
# city_list = [[30.378369, -97.763446, 30.313341, -97.706616]]
# city_list = [[30.357078, -97.813738, 30.259512, -97.665644]]
# city_list = [[30.387953, -97.843911, 30.249935, -97.635460]] # AS
# [-74.052914, -73.875168, 40.656702, 40.836357]
city_list = [[40.836357, -74.052914, 40.656702, -73.875168]]   # NY
# city_list = [[37.809524, -122.520352, 37.708991, -122.358712]]   # SA

# 划分网格之后数据分析
def cell_Analyse(checkins, index):
    checkins = checkins.reset_index()
    country_state = str(checkins.iloc[0]['country_id']) + " " + str(checkins.iloc[0]['state_id'] + " " + str(checkins.iloc[0]['city_id']))
    # print(country_state, len(checkins))
    print("用户数", len(checkins.uid.unique()))
    # print(checkins)
    save_city_checkins = checkins.ix[:, [1, 2, 3, 4, 5]]
    # print(save_city_checkins)
    path = "G:/pyfile/relation_protect/src/data/city_data/"
    # save_city_checkins.to_csv(path + str(checkins.iloc[0]['city_id']) + ".csv", sep='\t', header=True, index=False)
    save_city_checkins.to_csv(path + "NY.csv", sep='\t', header=True, index=False)
    return save_city_checkins
    # print(len(save_city_checkins))


def cell_Analyse1(checkins):
    checkins = checkins.reset_index()
    country_state = str(checkins.iloc[0]['country_id']) + " " + str(checkins.iloc[0]['state_id'])
    print(country_state, len(checkins))


def random_choose(locids):
    return np.random.choice(locids, int(len(locids) * 0.05), replace=False)


class data_handle():

    def __init__(self, ranges=None):
        self.ranges = ranges
        self.filename ="G:/pyfile/relation_protect/src/data/origin_data/Gowalla_totalCheckins.txt"
        # 格式为(maxlat,minlng),(minlat,maxlng) 为左上角点和右下角点

    # 过滤错误数据
    def get_all_dict(self):
        file = open(self.filename, "r")
        a = file.readline()
        count = 0
        while a:
            temp = a.strip().split("\t")
            if temp[0] != "" and temp[1] != "" and temp[2] != "" and temp[3] != "" and temp[4] != "":
               if temp[2].find(".") >= 0 and temp[3].find(".") >= 0:
                   if len(temp[2].split('.')[1]) >= 6 and len(temp[3].split('.')[1]) >= 6:
                      date = temp[1].split("T")
                      temp[1] = date[0] + " " + date[1].strip("Z")
                      dict.append(temp)
            count += 1
            a = file.readline()
        file.close()
        checkins = pd.DataFrame(dict, columns=['uid', 'time', 'latitude', 'longitude', 'locid'])
        print("数据清洗前总条数为：", count)
        print("清洗后数据条数为：", len(checkins))
        checkins.to_csv("G:/pyfile/relation_protect/src/data/origin_data/clean_totalCheckins.txt", header=True, sep="\t",index=False)
        return checkins

    # 基本数据分析
    def get_avg_userchenkins(self, checkins):
        print("总得签到记录数：", len(checkins))
        print("总用户数：", len(checkins.uid.unique()))
        print("总位置数：", len(checkins.locid.unique()))

        # 统计每个用户的总共签到次数
        usersCheckinTimes_uid = checkins.groupby("uid").size().reset_index(name="uid_times")
        # 每个用户的签到数据统计  最大 最小 平均
        usersCheckinTimesMax = np.max(usersCheckinTimes_uid.uid_times)
        usersCheckinTimesMin = np.min(usersCheckinTimes_uid.uid_times)
        usersCheckinTimesAvg = np.average(usersCheckinTimes_uid.uid_times)

        # 统计每个用户在签到过的地方的签到次数
        usersCheckinTImes_uid_locid = checkins.groupby(["uid", "locid"]).size().reset_index(name="uid_locid_times")
        usersCheckinTImes_uid_locidMax = np.max(usersCheckinTImes_uid_locid.uid_locid_times)
        usersCheckinTImes_uid_locidMin = np.min(usersCheckinTImes_uid_locid.uid_locid_times)
        usersCheckinTImes_uid_locidAvg = np.average(usersCheckinTImes_uid_locid.uid_locid_times)

        # 统计用户签到的不同位置个数
        usersCheckinDiversity = usersCheckinTImes_uid_locid.groupby(["uid"]).size().reset_index(name="uid_locid_uid_times")
        print("用户签到的不同位置个数小于等于2", len(usersCheckinDiversity[usersCheckinDiversity.uid_locid_uid_times <= 2]))
        usersCheckinDiversityMax = np.max(usersCheckinDiversity.uid_locid_uid_times)
        usersCheckinDiversityMin = np.min(usersCheckinDiversity.uid_locid_uid_times)
        usersCheckinDiversityAvg = np.average(usersCheckinDiversity.uid_locid_uid_times)

        # 位置签到数
        locid_nums = checkins.groupby("locid").size().reset_index(name="locid_times")
        print("位置签到个数小于等于1", len(locid_nums[locid_nums.locid_times <=1]))
        locCheckinTimesMax = np.max(locid_nums.locid_times)
        locCheckinTimesMin = np.min(locid_nums.locid_times)
        locCheckinTimesAvg = np.average(locid_nums.locid_times)

        print("位置签到数：", locCheckinTimesMax, locCheckinTimesMin, locCheckinTimesAvg)
        print("用户签到数：", usersCheckinTimesMax, usersCheckinTimesMin, usersCheckinTimesAvg)
        print("用户位置签到数：", usersCheckinTImes_uid_locidMax, usersCheckinTImes_uid_locidMin, usersCheckinTImes_uid_locidAvg)
        print("用户签到的不同位置个数：", usersCheckinDiversityMax, usersCheckinDiversityMin, usersCheckinDiversityAvg)

    def divide_area_by_range(self, checkin, index, maxlat, maxlng, minlat, minlng):
        latitude = decimal.Decimal(checkin[2])
        longitude = decimal.Decimal(checkin[3])
        if minlat <= latitude <= maxlat and minlng <= longitude <= maxlng:
            # 根据国家划分
            checkin.append(str(index))
            # for i in range(len(state_list)):
            #     if state_list[i][2] <= latitude <= state_list[i][0] and state_list[i][1] <= longitude <= state_list[i][3]:
            #         # 根据州区域划分
            #         checkin.append(str(i+1))
            #         # 州区域划分完成后进行划分,变成城市范围数据
            #         # checkin = self.divide_area_by_NN(checkin, N, i)\
            #         # 后期需要写成单独的函数
            #         for j in range(len(city_list)):
            #             if city_list[j][2] <= latitude <= city_list[j][0] and city_list[j][1] <= longitude <= \
            #                     city_list[j][3]:
            #                 # 根据城市进行划分
            #                 checkin.append(str(j + 1))
            #                 break
            #         break
            # 根据州划分
            self.divide_area_by_state0rcity(checkin, state_list)
            # 根据城市划分
            self.divide_area_by_state0rcity(checkin, city_list)
        return checkin

    def choose(self, checkins):
        print("开始筛选数据")
        usersCheckin = checkins.groupby(["uid"]).size().reset_index(name="uid_locid_uid_times")
        ins1 = usersCheckin[usersCheckin.uid_locid_uid_times >= 80]
        uids0 = ins1.uid.values

        # locid_nums = checkins.groupby("locid").size().reset_index(name="locid_times")
        # locids = locid_nums[locid_nums.locid_times >= 5]
        # print(len(locids))
        # # print("位置签到个数小于等于1", len(locid_nums[locid_nums.locid_times >= 5]))

        # 统计每个用户在签到过的地方的签到次数
        usersCheckinTImes_uid_locid = checkins.groupby(["uid", "locid"]).size().reset_index(name="uid_locid_times")
        # ins = usersCheckinTImes_uid_locid[(usersCheckinTImes_uid_locid.uid_locid_times >= 4) & (usersCheckinTImes_uid_locid.uid_locid_times <=12)]
        ins = usersCheckinTImes_uid_locid[(usersCheckinTImes_uid_locid.uid_locid_times >= 2)]
        uids = [row[0] for row in ins.itertuples(index=False, name=False)]
        locids = [[row[0], row[1]] for row in ins.itertuples(index=False, name=False)]
        print("用户位置签到数据为5~12的用户个数为", len(uids))
        locids = pd.DataFrame(locids, columns=['uid', 'locids'])

        # 统计用户签到过多少位置
        usersCheckinTImes_uid_locid = ins.groupby(["uid", "locid"]).size().reset_index(name="uid_locid_times")
        usersCheckinDiversity1 = usersCheckinTImes_uid_locid.groupby(["uid"]).size().reset_index(name="uid_locid_uid_times")
        # uids1 = usersCheckinDiversity1[(usersCheckinDiversity1.uid_locid_uid_times >= 15) & (usersCheckinDiversity1.uid_locid_uid_times <= 120)].uid.values
        uids1 = usersCheckinDiversity1[(usersCheckinDiversity1.uid_locid_uid_times >= 15)].uid.values
        print("用户签到的不同位置个数为10-40的用户个数为：", len(uids1))

        choose_uids = set(uids0).intersection(set(uids1)).intersection(set(uids))
        print("满足需求的用户个数为：", len(choose_uids))
        ano_checkins = []
        for row in checkins.itertuples(index=False, name=False):
            if row[0] in choose_uids:
                # random_locids = random_choose(checkins[checkins.uid == row[0]].locid.values)
                # choose_locids = set(locids[locids.uid == row[0]].locids.values).union(set(random_locids))
                choose_locids = set(locids[locids.uid == row[0]].locids.values)
                if row[4] in choose_locids:
                    checkin = list(row)
                    ano_checkins.append(checkin)
        ano_checkins = pd.DataFrame(ano_checkins, columns=['uid', 'time', 'latitude', 'longitude', 'locid'])
        ano_checkins.to_csv("G:/pyfile/relation_protect/src/data/city_data/FS_NY_1.csv", sep='\t', header=True, index=False)
        self.get_avg_userchenkins(ano_checkins)

    # 划分州或者城市
    def divide_area_by_state0rcity(self, checkin, city_list):
        latitude = decimal.Decimal(checkin[2])
        longitude = decimal.Decimal(checkin[3])
        for i in range(len(city_list)):
            if city_list[i][2] <= latitude <= city_list[i][0] and city_list[i][1] <= longitude <= city_list[i][3]:
                # 根据州区域划分
                checkin.append(str(i + 5))
                break

    # 整体网格划分步骤
    def divide_area(self):
        # 网格大小应该可以变化
        core_num = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(5)
        df = pool.starmap(self.divide_area_by_range, [(row, 1, 49.178086, -67.179640, 28.050016, -124.186610) for row in dict])
        pool.close()
        pool.join()
        checkins = pd.DataFrame(df, columns=['uid', 'time', 'latitude', 'longitude', 'locid', 'country_id', 'state_id', 'city_id'])
        # checkins.to_csv('G:/pyfile/relation_protect/src/data/origin_data/totalCheckins.csv', sep='\t', header=True, index=False)
        usercheckinConutry_state_city = checkins.groupby(["country_id", "state_id", "city_id"])
        Parallel(n_jobs=core_num)(delayed(cell_Analyse)(group[1], 2) for group in usercheckinConutry_state_city)

    def divide(self, checkin):
        latitude = decimal.Decimal(checkin[2])
        longitude = decimal.Decimal(checkin[3])
        ano_ckeckin = []
        for i in range(len(city_list)):
            if city_list[i][2] <= latitude <= city_list[i][0] and city_list[i][1] <= longitude <= city_list[i][3]:
                # 根据州区域划分
                ano_ckeckin = deepcopy(checkin)
                break
        return ano_ckeckin

    def divide_area_1(self):
        core_num = multiprocessing.cpu_count()
        checkins = pd.read_csv("G:/pyfile/relation_protect/src/data/origin_data/checkins.txt", delimiter="\t", index_col=None)
        checkins.columns = ['uid', 'time', 'time1', 'latitude', 'longitude', 'locid']
        checkins['time'] = checkins['time'].str.cat(checkins['time1'], sep=" ")
        print(checkins.time.values)
        checkins.drop(columns=['time1'], inplace=True)
        df = checkins[(checkins.latitude >= city_list[0][2]) & (checkins.latitude <= city_list[0][0]) & (checkins.longitude >= city_list[0][1]) & (checkins.longitude <= city_list[0][3])]
        # df = Parallel(n_jobs=core_num)(delayed(self.divide)(row for row in checkins.itertuples(index=False,name=False)))
        # checkins = pd.DataFrame(df, columns=['uid', 'time', 'latitude', 'longitude', 'locid'])
        path = "G:/pyfile/relation_protect/src/data/city_data/"
        df.to_csv(path + "SNAP_NY.csv", sep='\t', header=True, index=False)


    def divide_area_2(self):
        checkins = pd.read_csv("G:/pyfile/relation_protect/src/data/origin_data/dataset_TSMC2014_NYC.txt", delimiter="\t", header=None, encoding='ISO-8859-1')
        checkins.columns = ['uid', 'locid', 'catid', 'catname', 'latitude', 'longitude', 'timezone', 'time']
        # checkins.columns = ['uid', 'locid', 'catid', 'catname', 'timezone', 'time']
        # print(checkins)
        checkins.loc[:, 'time'] = checkins['time'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(x, '%a %b %d %H:%M:%S %z %Y')))
        choose_checkins = checkins.ix[:, [0, 7, 4, 5, 1]]
        df = choose_checkins[(choose_checkins.latitude >= city_list[0][2]) & (choose_checkins.latitude <= city_list[0][0]) & (choose_checkins.longitude >= city_list[0][1]) & (choose_checkins.longitude <= city_list[0][3])]
        path = "G:/pyfile/relation_protect/src/data/city_data/"
        df.to_csv(path + "FS_NY.csv", sep='\t', header=True, index=False, encoding='ISO-8859-1')


if __name__ == "__main__":

    start = time.time()
    data_handler = data_handle()
#     checkins = data_handler.get_all_dict()
#     checkins = pd.read_csv("G:/pyfile/relation_protect/src/data/origin_data/dataset_TSMC2014_NYC.txt", delimiter="\t", index_col=None, encoding="utf_8")
    # data_handler.get_avg_userchenkins(checkins)
    # data_handler.divide_area_1()
    # data_handler.divide_area_2()
    checkins1 = pd.read_csv("G:/pyfile/relation_protect/src/data/city_data/FS_NY.csv", delimiter="\t", index_col=None)
    data_handler.choose(checkins1)
    # checkins1 = pd.read_csv("G:/pyfile/relation_protect/src/data/city_data/1.csv", delimiter="\t", index_col=None)
    # data_handler.get_avg_userchenkins(checkins1)
    end = time.time()
    duration = str(end - start)
    print("总花费时间为：", duration)



