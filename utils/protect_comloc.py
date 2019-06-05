#!/usr/bin/env python
# encoding: utf-8


import pandas as pd
from joblib import Parallel, delayed
import multiprocessing as mp
from itertools import combinations
import math
import gc
import numpy as np
from copy import deepcopy
import time


class protect_comloc:

    def __init__(self, user_num, tablename):
        # self.ano_checkins = []
        self.user_num = user_num
        self.ano_checkins_tablename = tablename
        pass

    def k_freq_loc(self, u, u_checkins, k):
        u_loc = pd.DataFrame(u_checkins['locid'].value_counts()).reset_index()  # 统计locid的不同值及其个数
        u_loc.columns = ["locid", "cnt"]
        if k <= len(u_loc):
            u_loc = u_loc[0:k]
        return [u, list(u_loc.locid.values)]

    def u_comloc(self, u1_checkins, u2_checkins, u1, u2):
        u1_locids = u1_checkins.locid.unique()
        u2_locids = u2_checkins.locid.unique()
        comloc = list(set(u1_locids).intersection(set(u2_locids)))
        return [u1, u2, len(comloc), comloc]

    def comloc_protection(self, checkins, k, m):  # k为用户的前k个频繁访问位置、m为前m%的共同访问位置，checkins是签到数据
        uids = checkins.uid.unique()
        checkin1 = checkins.groupby(by=['locid', 'latitude', 'longitude']).size().reset_index(name="locid_time")
        locids = [row[0] for row in checkin1.itertuples(index=False, name=False)]  # 记录locid对应的经纬度，以便在替换locid时将相应的位置数据也进行替换
        lat_lon = [[row[1], row[2]] for row in checkin1.itertuples(index=False, name=False)]
        del checkin1    # 释放checkin1的内存空间
        gc.collect()
        core_num = mp.cpu_count()
        # print(checkins)
        # print(uids)
        checkins = checkins.sort_values(by=['uid'], ascending=True).reset_index(drop=True)  # uid的降序排列
        user_k_freqloc = Parallel(n_jobs=core_num)(delayed(self.k_freq_loc)(u, checkins[checkins.uid == u], k) for u in uids)
        user_k_freqloc = pd.DataFrame(user_k_freqloc, columns=['uid', 'k_freqlocs'])
        # print(user_k_freqloc)
        pairs = pd.DataFrame(list(combinations(uids, 2)), columns=['u1', 'u2'])
        pairs_comloc = Parallel(n_jobs=core_num)(delayed(self.u_comloc)(checkins[checkins.uid == row[0]], checkins[checkins.uid == row[1]], row[0], row[1]) for row in pairs.itertuples(index=False, name=False))
        pairs_comloc = pd.DataFrame(pairs_comloc, columns=["u1", "u2", "comloc_num", "comlocs"])
        nums = int(math.ceil(len(pairs_comloc) * m))
        pairs_comloc = pairs_comloc.sort_values(by='comloc_num', ascending=False).reset_index(drop=True)
        pairs_comloc = pairs_comloc[0:nums]   # 20%的共同访问位置用户对,即使需要保持的共同访问位置的用户对
        # print(pairs_comloc)
        protect_uids = set(pairs_comloc.u1.values).union(set(pairs_comloc.u2.values))  # 需要进行共同访问位置保护的用户id
        disturb_uid = set(uids) - protect_uids     # 不需要进行共同访问位置保护的用户
        # print(disturb_uid)
        ano_checkins = []
        for u in disturb_uid:   # 对于没有需要保护共同访问位置的用户，对签到数据进行随机扰动
            u_checkins = deepcopy(checkins[checkins.uid == u])
            u_locids = u_checkins.locid.unique()
            for row in u_checkins.itertuples(index=False, name=False):
                # locid = np.random.choice(locids)
                locid = np.random.choice(u_locids)
                u_lat_lon = lat_lon[locids.index(locid)]
                lat = u_lat_lon[0]
                lng = u_lat_lon[1]
                # ano_checkin = [row[0], row[1], lat, lng, locid, row[5]]
                ano_checkin = [row[0], row[1], lat, lng, locid]
                ano_checkins.append(ano_checkin)
        for u in protect_uids:   # 对于需要保护共同访问位置的用户，除了共同访问位置中的非频繁访问位置之外的所有位置进行扰动
            u_checkins = deepcopy(checkins[checkins.uid == u])
            u_locids = u_checkins.locid.unique()
            comlocs = pairs_comloc[(pairs_comloc.u1 == u) | (pairs_comloc.u2 == u)].comlocs
            u_comlocs = []
            list(map(lambda x: u_comlocs.extend(x), comlocs))
            u_comlocs = set(u_comlocs)
            u_freqloc = user_k_freqloc[user_k_freqloc.uid == u].k_freqlocs.values[0]
            u_protect_locs = set(u_comlocs) - set(u_freqloc)  # 共同访问位置和频繁访问位置的差集
            # if (len(set(u_comlocs) - set(locids)) == 0) & (len(set(u_comlocs)) == len(set(locids))):
            #     u_distrublocs = list(set(locids))
            if len(set.union(set(u_comlocs), set(u_freqloc))) == len(u_locids):
                u_distrublocs = list(set(u_locids))
            # if (len(set(u_comlocs) - set(u_locids)) == 0) & (len(set(u_comlocs)) == len(set(u_locids))):
            #     u_distrublocs = list(set(u_locids))
            elif len(set(u_comlocs).intersection(set(u_freqloc))) == 0:
                u_distrublocs = list(set(u_locids))
            else:
                u_distrublocs = list(set(u_locids) - u_protect_locs)     # 可以选择的locid
            for row in u_checkins.itertuples(index=False, name=False):
                if row[4] not in u_protect_locs:
                    locid = np.random.choice(u_distrublocs)
                    u_lat_lon = lat_lon[locids.index(locid)]
                    lat = u_lat_lon[0]
                    lng = u_lat_lon[1]
                    # ano_checkin = [row[0], row[1], lat, lng, locid, row[5]]
                    ano_checkin = [row[0], row[1], lat, lng, locid]
                    ano_checkins.append(ano_checkin)
                else:
                    ano_checkins.append(list(row))
        self.save_ano_checkins(ano_checkins)
        # return ano_checkins  # 扰动后的匿名数据

    def save_ano_checkins(self, ano_checkin):
        ano_checkins = pd.DataFrame(ano_checkin)
        ano_checkins.to_csv("G:/pyfile/relation_protect/src/data/result_data/" + self.ano_checkins_tablename + "/3_" +
                            str(self.user_num) + "_" + self.ano_checkins_tablename + ".csv", header=None, index=None, sep='\t', mode='a')

    def comnunity_disturb(self, checkins, k, m):
        community_checkins = checkins.groupby(["clusterid"])
        print(len(community_checkins))
        checkin_cnt = 0
        checkin_chunk_size = math.ceil(len(community_checkins) / 10)
        for group in community_checkins:
            if checkin_cnt % checkin_chunk_size == 0:  # finished the anonymization of a chunk of checkins打印一部分匿名化的结果
                print('%-3d%% work complete.' % (int(checkin_cnt / checkin_chunk_size) * 10))
            self.comloc_protection(group[1], k, m)
            checkin_cnt += 1
        # 将数据按照uid升序进行排列
        ano_checkins = pd.read_csv("G:/pyfile/relation_protect/src/data/result_data/" + self.ano_checkins_tablename + "/3_" +
                                   str(self.user_num) + "_" + self.ano_checkins_tablename + ".csv", index_col=None,
                                   sep='\t', names=['uid', 'times', 'latitude', 'longitude', 'locid', 'clusterid'])
        ano_checkins = ano_checkins.sort_values(by=['uid']).reset_index(drop=True)
        ano_checkins.to_csv("G:/pyfile/relation_protect/src/data/result_data/" + self.ano_checkins_tablename + "/3_" +
                            str(self.user_num) + "_" + self.ano_checkins_tablename + ".csv", header=None, index=None, sep='\t')

if __name__ == "__main__":

    # for k in [3, 4, 5, 6, 7, 8, 9, 10]:
    for k in [3]:
        start = time.time()
        pc = protect_comloc(int(k), "comloc")
        checkins = pd.read_csv("G:/pyfile/relation_protect/src/data/city_data/test.csv", delimiter="\t", index_col=None)
        # checkins = pd.read_csv(
        #     "G:/pyfile/relation_protect/src/data/result_data/1_comloc_" + str(k) + "_user_simple_community.data",
        #     delimiter="\t", names=["uid", "time", "latitude", "longitude", "locid", "clusterid"], header=None)
        # pc.comnunity_disturb(checkins, 3, 0.6)

        pc.comloc_protection(checkins, 3, 0.2)
        end = time.time()
        print("花费时间：", str(end-start))
        # data = pd.DataFrame(data, columns=["uid", "time", "lat", "lng", "locid"])
        # print(data)
        # data.to_csv("G:/pyfile/relation_protect/src/data/city_data/test1.csv", sep='\t', index=False, header=False)
