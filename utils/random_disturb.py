#!/usr/bin/env python  
# encoding: utf-8  

""" 
@version: v1.0 
@author: jaculamata 
@license: Apache Licence  
@contact: 819436557@qq.com 
@site: http://blog.csdn.net/hqzxsc2006 
@software: PyCharm 
@file: 7_contrast.py 
@time: 2019/1/20 15:45 
"""
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import time
from copy import deepcopy
from sys import  argv
result_path = "G:/pyfile/relation_protect/src/data/result_data/"
city_path = "G:/pyfile/relation_protect/src/data/city_data/"
from random_disturb_security import random_disturb_security

class random_disturb():
    def __init__(self):
        self.replace_uids = []
        self.replace_locids = []
        self.ano_checkins = []
        self.locids = []
        self.lat_lon = []
        pass

    def raplace_locids(self, u_checkins, ratio):
        # replace_chenkinsnum = int(math.ceil(u_checkins.shape[0] * ratio))
        replace_chenkinsnum = int(u_checkins.shape[0] * ratio)
        replace_locid = np.random.choice(u_checkins.locid, replace_chenkinsnum, replace=False)
        # if replace_chenkinsnum > len(u_checkins.locid.unique()):
        #     replace_locid = np.random.choice(u_checkins.locid.unique(), replace_chenkinsnum, replace=True)
        #     replace_locid = list(set(replace_locid))
        # else:
        #     replace_locid = list(np.random.choice(u_checkins.locid.unique(), replace_chenkinsnum, replace=False))
        return [u_checkins.uid.unique()[0], replace_locid]

    def locid_to_latlon(self, checkins):   # 可以改成[]的形式，不用append
        checkins = checkins.groupby(by=['locid', 'latitude', 'longitude']).size().reset_index(name="locid_time")
        for row in checkins.itertuples(index=False, name=False):
            self.locids.append(row[0])
            self.lat_lon.append([row[1], row[2]])

    def random_disturb1(self, city, path, ratio, defence_method, time):
        start = time.time()
        checkins = pd.read_csv(path + city + '.csv', delimiter="\t", index_col=None)
        self.locid_to_latlon(checkins)
        locids = checkins.locid.unique()
        checkin_cnt = 0
        checkin_chunk_size = int(len(checkins.uid.unique()) / 10)
        checkinid = 0
        for u in checkins.uid.unique():
            checkinid += 1
            if checkin_cnt % checkin_chunk_size == 0:  # finished the anonymization of a chunk of checkins打印一部分匿名化的结果
                print('%-3d%% work complete.' % (int(checkin_cnt / checkin_chunk_size) * 10))
            u_checkins = checkins[checkins.uid == u]
            sample = deepcopy(u_checkins.sample(frac=ratio, replace=False))
            all_index = sample.index
            u_locids = sample.locid.unique()
            for checkin in u_checkins.itertuples(index=True, name=False):
                ano_checkin = [checkin[1], checkin[2], checkin[3], checkin[4], checkin[5]]
                if checkin[0] in all_index:
                    # u_locids = checkins[checkins.uid == checkin[1]].locid.unique()
                    replace_locids = list(set(locids) - (set(u_locids)))
                    replace_locid = np.random.choice(replace_locids)
                    lat_lon = self.lat_lon[self.locids.index(replace_locid)]
                    ano_checkin[2] = lat_lon[0]
                    ano_checkin[3] = lat_lon[1]
                    ano_checkin[4] = replace_locid
                self.ano_checkins.append(ano_checkin)
            checkin_cnt += 1
        ano_checkins = pd.DataFrame(self.ano_checkins, columns=['uid', 'time', 'latitude', 'longitude', 'locid'])
        ano_checkins.to_csv(result_path + "random_disturb/" + city + "_" + str(ratio) + '_' + defence_method + '_' + time + '.checkin',sep="\t", header=None, index=False)
        end = time.time()
        print("花费的时间为：", str(end - start))
        # random_disturb = random_disturb_security(ano_checkins, city, 40, 30, [30.387953, -97.843911, 30.249935, -97.635460], 0.004492 * 2, 0.005202 * 2)  # AS
        # random_disturb = random_disturb_security(ano_checkins, city, 40, 30, [37.809524, -122.520352, 37.708991, -122.358712], 0.004492 * 2, 0.005681 * 2) #SF
        random_disturb = random_disturb_security(ano_checkins, city, 40, 40, [40.836357, -74.052914, 40.656702, -73.875168], 0.0044966 * 2, 0.0059352 * 2)  # FS_NY_1
        random_disturb.cal_security(ratio)
        return ano_checkins

    def random_disturb2(self, city, path, ratio, defence_method, i):
        start = time.time()
        checkins = pd.read_csv(path + city + '.csv', delimiter="\t", index_col=None)
        self.locid_to_latlon(checkins)
        locids = checkins.locid.unique()
        checkin_cnt = 0
        checkin_chunk_size = int(len(checkins.uid.unique()) / 10)
        checkinid = 0
        for u in checkins.uid.unique():
            checkinid += 1
            if checkin_cnt % checkin_chunk_size == 0:  # finished the anonymization of a chunk of checkins打印一部分匿名化的结果
                print('%-3d%% work complete.' % (int(checkin_cnt / checkin_chunk_size) * 10))
            u_checkins = checkins[checkins.uid == u]
            sample = deepcopy(u_checkins.sample(frac=ratio, replace=False))
            all_index = sample.index
            u_locids = sample.locid.unique()
            for checkin in u_checkins.itertuples(index=True, name=False):
                ano_checkin = [checkin[0], checkin[1], checkin[2], checkin[3], checkin[4], checkin[5]]
                if checkin[0] in all_index:
                    # u_locids = checkins[checkins.uid == checkin[1]].locid.unique()
                    replace_locids = list(set(locids) - (set(u_locids)))
                    replace_locid = np.random.choice(replace_locids)
                    lat_lon = self.lat_lon[self.locids.index(replace_locid)]
                    ano_checkin[3] = lat_lon[0]
                    ano_checkin[4] = lat_lon[1]
                    ano_checkin[5] = replace_locid
                self.ano_checkins.append(ano_checkin)
            checkin_cnt += 1
        ano_checkins = pd.DataFrame(self.ano_checkins, columns=['index', 'uid', 'time', 'latitude', 'longitude', 'locid'])
        ano_checkins = ano_checkins.set_index("index").sort_index()
        ano_checkins.to_csv(result_path + "random_disturb/" + city + "_" + str(ratio) + '_' + defence_method + '_' + str(i) + '.checkin',sep="\t", header=None, index=False)
        end = time.time()
        print("花费的时间为：", str(end - start))
        # random_disturb = random_disturb_security(ano_checkins, city, 40, 30, [30.387953, -97.843911, 30.249935, -97.635460], 0.004492 * 2, 0.005202 * 2)  # AS
        # random_disturb = random_disturb_security(ano_checkins, city, 40, 30, [37.809524, -122.520352, 37.708991, -122.358712], 0.004492 * 2, 0.005681 * 2) #SF
        random_disturb = random_disturb_security(ano_checkins, city, 40, 40, [40.836357, -74.052914, 40.656702, -73.875168], 0.0044966 * 2, 0.0059352 * 2)  # FS_NY_1
        random_disturb.cal_security(ratio)
        return ano_checkins

    def random_disturb(self, city, path, ratio1, defence_method):
        start = time.time()
        checkins = pd.read_csv(path + city + '.csv', delimiter="\t", index_col=None)
        self.locid_to_latlon(checkins)
        uids, locids = checkins.uid.unique(), checkins.locid.unique()
        # ratio1 = ratio/100
        # print(ratio1)

        # 1. 在原始数据中选取一部分记录进行随机扰动
        core_num = multiprocessing.cpu_count()
        replace_checkin = Parallel(n_jobs=core_num)(delayed(self.raplace_locids)(checkins.loc[checkins.uid == u], ratio1) for u in uids)
        replace_checkin = pd.DataFrame(replace_checkin, columns=['uid', 'replace_locids'])
        # print(len(replace_checkin))
        self.replace_uids = [row[0] for row in replace_checkin.itertuples(name=False, index=False)]
        self.replace_locids = [row[1] for row in replace_checkin.itertuples(name=False, index=False)]

        # 2 Begin to anonymize all checkins.匿名化签到数据
        checkin_cnt = 0
        checkin_chunk_size = int(len(checkins) / 10)
        checkinid = 0
        for row in checkins.itertuples(index=False, name=False):
            checkinid += 1
            if checkin_cnt % checkin_chunk_size == 0:  # finished the anonymization of a chunk of checkins打印一部分匿名化的结果
                print('%-3d%% work complete.' % (int(checkin_cnt / checkin_chunk_size) * 10))
            uid, locid = row[0], row[4]
            ano_checkin = [uid, row[1]]
            if uid in self.replace_uids:
                u_locids = self.replace_locids[self.replace_uids.index(uid)]
                if locid in u_locids:
                    replace_locids = list(set(locids) - (set(u_locids)))
                    replace_locid = np.random.choice(replace_locids)
                    lat_lon = self.lat_lon[self.locids.index(replace_locid)]
                    ano_checkin.append(lat_lon[0])
                    ano_checkin.append(lat_lon[1])
                    ano_checkin.append(replace_locid)
                    ano_checkin.append(1)
                else:
                    ano_checkin.append(row[2])
                    ano_checkin.append(row[3])
                    ano_checkin.append(row[4])
                    ano_checkin.append(0)
            else:
                ano_checkin.append(row[2])
                ano_checkin.append(row[3])
                ano_checkin.append(row[4])
                ano_checkin.append(0)
            self.ano_checkins.append(ano_checkin)
            checkin_cnt += 1

        checkins = pd.DataFrame(self.ano_checkins, columns=['uid', 'time', 'latitude', 'longitude', 'locid', 'num'])
        sum = checkins['num'].sum()
        # print("ratio", sum * 1.0/len(checkins))
        # checkins.locid.apply(float).apply(int)
        checkins.to_csv(result_path + "random_disturb/" + city + "_" + str(ratio1) + '_' + defence_method + '.checkin', sep="\t", header=None, index=False)
        end = time.time()
        print("花费的时间为：", str(end-start))
        random_disturb = random_disturb_security(checkins, city, 40, 30, [30.387953, -97.843911, 30.249935, -97.635460],0.004492 * 2, 0.005202 * 2)  # AS
        # random_disturb = random_disturb_security(checkins, city, 40, 30, [37.809524, -122.520352, 37.708991, -122.358712], 0.004492 * 2, 0.005681 * 2) #SF
        random_disturb.cal_security(3)
        return checkins


if __name__ == "__main__":
    start = time.time()
    # r=float(argv[1])
    # for r in [0.2538, 0.3277, 0.3481, 0.3758, 0.3946, 0.4584, 0.4620]:
    # for r in [0.1522, 0.1773, 0.1763, 0.2821, 0.2757, 0.3006, 0.3137]:
    for r in [0.1123, 0.1283, 0.1265, 0.1443, 0.1780, 0.1997, 0.2095]:
    # for i in [2, 4, 5.5, 7, 9, 12, 17, 21]: # 24  42 47  50  58  63  72  79
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            contrast = random_disturb()
            checkin_obf = pd.read_csv(city_path + "FS_NY_1"+".csv", delimiter="\t", index_col=None)
            contrast.locid_to_latlon(checkin_obf)
    # checkin = contrast.random_disturb("SF", city_path, 1.2, "random_disturb")
    # checkin = contrast.random_disturb("SF", city_path, 3.8, "random_disturb")
    # checkin = contrast.random_disturb("FS_NY_1", city_path, 4, "random_disturb")  # 30
    # checkin = contrast.random_disturb("FS_NY_1", city_path, 2.5, "random_disturb")  # 20
    # checkin = contrast.random_disturb("FS_NY_1", city_path, 1.3, "random_disturb")  # 10
    # checkin = contrast.random_disturb("FS_NY_1", city_path, 5.5, "random_disturb")  # 40
    # checkin = contrast.random_disturb("FS_NY_1", city_path, 7.2, "random_disturb")  # 50
    # checkin = contrast.random_disturb("FS_NY_1", city_path, 9.2, "random_disturb")  # 60
    # checkin = contrast.random_disturb("FS_NY_1", city_path, 13, "random_disturb")  # 70
            checkin = contrast.random_disturb2("FS_NY_1", city_path, r, "random_disturb", i)  # 0


    # checkin = contrast.random_disturb("1", city_path, 3, "random_disturb")
    # checkin = contrast.random_disturb("1", city_path, 4.3, "random_disturb")
    # checkin = contrast.random_disturb("1", city_path, 5.5, "random_disturb")
    # checkin = contrast.random_disturb("1", city_path, 7, "random_disturb")
    # checkin = contrast.random_disturb("1", city_path, 9, "random_disturb")
    # checkin = contrast.random_disturb("1", city_path, 12, "random_disturb")
    # checkin = contrast.random_disturb("1", city_path, 17, "random_disturb")
