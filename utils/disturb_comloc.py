#!/usr/bin/env python
# encoding: utf-8


import pandas as pd
from joblib import Parallel, delayed
import multiprocessing as mp
from itertools import combinations
import math
import gc
import numpy as np
from itertools import repeat
from copy import deepcopy
import time
from math import cos, sin, atan2, sqrt, radians, degrees


class disturb_comloc:

    def __init__(self, user_num, tablename, q, path, lons_per_km, lats_per_km, city):  # q是共同访问位置相似度
        self.user_num = user_num
        self.ano_checkins_tablename = tablename
        self.golbal_freqlocs = []
        self.lons_per_km = lons_per_km  # delta longitudes per kilo meter
        self.lats_per_km = lats_per_km  # delta latitudes per kilo meter
        self.comloc_num = int(math.ceil(-1/math.log(q)))
        self.path = path
        self.city = city
        self.visitedCluster = []
        self.add_dummy_user = []

    def k_freq_loc(self, u, u_checkins, k):
        u_loc = pd.DataFrame(u_checkins['locid'].value_counts()).reset_index()  # 统计locid的不同值及其个数
        u_loc.columns = ["locid", "cnt"]
        u_loc = u_loc.sort_values(by=['cnt', 'locid'], ascending=False).reset_index(drop=True)
        if k <= len(u_loc):
            u_loc = u_loc[0:k]
        u_locids = [row[0] for row in u_loc.itertuples(index=False,name=False)]
        u_loc_times = [row[1] for row in u_loc.itertuples(index=False,name=False)]
        return [u, u_locids, u_loc_times]

    def u_comloc(self, u1_checkins, u2_checkins, u1, u2):
        u1_locids = u1_checkins.locid.unique()
        u2_locids = u2_checkins.locid.unique()
        comloc = list(set(u1_locids).intersection(set(u2_locids)))
        return [u1, u2, len(comloc), comloc]

    # 两点之间的欧氏距离计算
    def euclidean_distance(self, loc1, loc2):
        return math.sqrt(((loc1[1] - loc2[1]) / self.lons_per_km) ** 2 + ((loc1[0] - loc2[0]) / self.lats_per_km) ** 2)

    # 计算该点是否可达，时间和空间
    def reach_point(self, checkins, timestamp_list, loc1):
        checkins = checkins.sort_values(by=['timestamp'], ascending=False).reset_index(drop=True)
        for i in range(len(timestamp_list)):
            checkins_before = checkins[checkins.timestamp >= timestamp_list[i]]
            checkins_after = checkins[checkins.timestamp < timestamp_list[i]]
            speed1, speed2 = -1, -1
            if len(checkins_before) > 0:
                checkin_before = list(checkins_before.values[len(checkins_before) - 1])
                if timestamp_list[i] == checkin_before[8]:
                    continue
                speed1 = self.euclidean_distance(loc1, [checkin_before[2], checkin_before[3]]) / (abs(timestamp_list[i] - checkin_before[8])/3600)
            if len(checkins_after) > 0:
                checkin_after = list(checkins[checkins.timestamp < timestamp_list[i]].values[0])
                speed2 = self.euclidean_distance(loc1, [checkin_after[2], checkin_after[3]]) / (abs(timestamp_list[i] - checkin_after[8])/3600)
            if (speed1 <= 60) & (speed2 <= 60):
                return i
        return -1

    def is_replace_locid(self, checkins, timestamp, loc1):
        if len(checkins) == 0:
            return True
        else:
            checkins = checkins.sort_values(by=['timestamp'], ascending=False).reset_index(drop=True)
            checkins_before = checkins[checkins.timestamp >= timestamp]
            checkins_after = checkins[checkins.timestamp < timestamp]
            speed1, speed2 = -1, -1
            if len(checkins_before) > 0:
                checkin_before = list(checkins_before.values[len(checkins_before) - 1])
                if timestamp == checkin_before[8]:
                    return False
                speed1 = self.euclidean_distance(loc1, [checkin_before[2], checkin_before[3]]) / (abs(timestamp - checkin_before[8])/3600)
            if len(checkins_after) > 0:
                checkin_after = list(checkins[checkins.timestamp < timestamp].values[0])
                speed2 = self.euclidean_distance(loc1, [checkin_after[2], checkin_after[3]]) / (abs(timestamp - checkin_after[8])/3600)
            if (speed1 <= 60) & (speed2 <= 60):
                return True
            return False

    def distance_sort(self, cdt_loc, ano_locs, locids):    # 获得ano_locs中所有位置与指定位置的距离
        dis_list = list(map(self.euclidean_distance, ano_locs, repeat(cdt_loc)))
        return [locids, dis_list]

    def center_geolocation(self, geolocations):
        x = 0
        y = 0
        z = 0
        lenth = len(geolocations)
        for lat, lon in geolocations:
            lon = radians(float(lon))
            lat = radians(float(lat))
            x += cos(lat) * cos(lon)
            y += cos(lat) * sin(lon)
            z += sin(lat)
        x = float(x / lenth)
        y = float(y / lenth)
        z = float(z / lenth)
        return [degrees(atan2(z, sqrt(x * x + y * y))), degrees(atan2(y, x))]

    def unsim_comloc_disturb(self, checkins, k):  # k为用户的前k个频繁访问位置、m为前m%的共同访问位置，checkins是签到数据

        uids = checkins.uid.unique()
        core_user = checkins[checkins.is_core == 1].uid.values[0]  # 社区内的核心用户
        clusterId = checkins.clusterid.values[0]  # 社区id
        self.visitedCluster.append(clusterId)
        community_locs = list(checkins.locid.unique())
        core_num = mp.cpu_count()
        checkins = checkins.sort_values(by=['uid'], ascending=True).reset_index(drop=True)  # uid的降序排列
        user_k_freqloc = Parallel(n_jobs=core_num)(delayed(self.k_freq_loc)(u, checkins[checkins.uid == u], k) for u in uids)
        user_k_freqloc = pd.DataFrame(user_k_freqloc, columns=['uid', 'k_freqlocs', 'freqloc_nums'])
        pairs = pd.DataFrame(list(combinations(uids, 2)), columns=['u1', 'u2'])
        pairs_comloc = Parallel(n_jobs=core_num)(delayed(self.u_comloc)(checkins[checkins.uid == row[0]], checkins[checkins.uid == row[1]], row[0], row[1]) for row in pairs.itertuples(index=False, name=False))
        pairs_comloc = pd.DataFrame(pairs_comloc, columns=["u1", "u2", "comloc_num", "comlocs"])
        pairs_comloc = pairs_comloc.sort_values(by='comloc_num', ascending=False).reset_index(drop=True)
        uids = list(uids)
        ano_checkins = []
        community_unfreq_uncomlocs = []
        u_disturb_locs = []
        u_freq_uncomlocs = []
        u_choose_locs = []
        community_u_comlocs = []
        community_comloc = []
        u_unfreq_uncomloc_list = []
        community_freq_uncomloc = []
        for u in uids:            # 获取社区内每个用户的非频繁和非共同访问位置,index与uid 是对应的
            u_checkins = deepcopy(checkins[checkins.uid == u])
            u_locids = u_checkins.locid.unique()   # 用户的所有访问位置
            comlocs = pairs_comloc[(pairs_comloc.u1 == u) | (pairs_comloc.u2 == u)].comlocs
            u_comlocs = []                         # 用户的共同访问位置
            list(map(lambda x: u_comlocs.extend(x), comlocs))
            u_comlocs = list(set(u_comlocs))
            if len(u_comlocs) > len(community_u_comlocs):
                community_u_comlocs = u_comlocs
            community_comloc.extend(u_comlocs)
            u_freqloc = user_k_freqloc[user_k_freqloc.uid == u].k_freqlocs.values[0]    # 用户的频繁访问位置
            u_unfreq_uncomlocs = list(set(u_locids) - set(u_freqloc) - set(u_comlocs))  # 用户的非频繁和非共同访问位置
            community_unfreq_uncomlocs.extend(u_unfreq_uncomlocs)  # 只有一个列表
            u_unfreq_uncomloc_list.append(u_unfreq_uncomlocs)      # 用户的非频繁非共同访问位置，是个二维列表
            u_choose_locs.append(u_unfreq_uncomlocs)
            community_freq_uncomloc.extend(list(set(u_freqloc) - set(u_comlocs)))
            u_freq_uncomlocs.append(list(set(u_freqloc) - set(u_comlocs)))  # 用户频繁访问位置中的非共同访问位置
            u_disturb_locs.append(list(set(u_comlocs) - set(u_freqloc)))  # 共同访问位置中的非频繁访问位置，即需要扰动的共同访问位置
        community_unfreq_uncomlocs = list(set(community_unfreq_uncomlocs))  # list 去重
        core_user_latlon = [self.lat_lon[self.locids.index(loc)] for loc in list(checkins[checkins.uid == core_user].locid.unique())]  # 获得核心用户的所有位置的经纬度
        core_user_centerpoint = self.center_geolocation(core_user_latlon)   # 获得核心用户的中心点
        if len(community_unfreq_uncomlocs) >= self.comloc_num:
            # 按照距离/**
            # dict_loc_dis = self.distance_sort(core_user_centerpoint, [self.lat_lon[self.locids.index(loc)] for loc in community_unfreq_uncomlocs], community_unfreq_uncomlocs)
            # dict_loc_dis = list(map(list, zip(*dict_loc_dis)))
            # dict_loc_dis = pd.DataFrame(dict_loc_dis, columns=['locid', 'dis']).sort_values(by=['dis'], ascending=True).reset_index(drop=True)  # 按照距离升序排列
            # if len(community_unfreq_uncomlocs) >= self.comloc_num*2:
            #     dict_loc_dis = dict_loc_dis[0:(self.comloc_num*2)]
            # community_unfreq_uncomlocs = np.random.choice(list(dict_loc_dis.locid.unique()), self.comloc_num, replace=False)**/
            # 按照距离和频率被删了
            community_unfreq_uncomlocs = np.random.choice(community_unfreq_uncomlocs, self.comloc_num, replace=False)
        else:
            # 按照距离/**
            # dict_loc_dis = self.distance_sort(core_user_centerpoint, [self.lat_lon[self.locids.index(loc)] for loc in list(set(self.golbal_freqlocs)-set(community_locs))], list(set(self.golbal_freqlocs)-set(community_locs)))
            # # dict_loc_dis = {'locid': community_unfreq_uncomlocs, 'dis': loc_dis}
            # dict_loc_dis = list(map(list, zip(*dict_loc_dis)))
            # dict_loc_dis = pd.DataFrame(dict_loc_dis, columns=['locid', 'dis']).sort_values(by=['dis'], ascending=True).reset_index(drop=True)  # 按照距离升序排列
            # dict_loc_dis = dict_loc_dis[0:(self.comloc_num*2)]
            # add_loc = np.random.choice(list(dict_loc_dis.locid.unique()), self.comloc_num - len(community_unfreq_uncomlocs), replace=False)
            # community_unfreq_uncomlocs.extend(add_loc)**/
            # 按照距离和频率被删了
            while len(community_unfreq_uncomlocs) < self.comloc_num:
                locid = np.random.choice(self.golbal_freqlocs)
                if (locid not in community_unfreq_uncomlocs) & (locid not in community_locs):
                    community_unfreq_uncomlocs.append(locid)
        # 最新修改/**
        # loc_freq_nums = [[loc, float(len(self.checkins[self.checkins.locid == loc]))/len(self.checkins)] for loc in community_unfreq_uncomlocs] # 记录社区中用户的频繁访问位置的访问频率
        # if len(community_unfreq_uncomlocs) >= self.comloc_num:
        #     dict_loc_dis = self.distance_sort(core_user_centerpoint, [self.lat_lon[self.locids.index(loc)] for loc in community_unfreq_uncomlocs], community_unfreq_uncomlocs)
        #     dict_loc_dis = list(map(list, zip(*dict_loc_dis)))
        #     for i in range(len(dict_loc_dis)):
        #         dict_loc_dis[i][1] = math.exp(-dict_loc_dis[i][1])/loc_freq_nums[i][1] * 100
        #     dict_loc_dis = pd.DataFrame(dict_loc_dis, columns=['locid', 'weight']).sort_values(by=['weight'], ascending=False).reset_index(drop=True)  # 按照距离升序排列
        #     sum = dict_loc_dis.weight.sum()
        #     dict_loc_dis.loc[:, 'weight'] = dict_loc_dis.loc[:, 'weight']/sum
        #     loc = [row[0] for row in dict_loc_dis.itertuples(index=False, name=False)]
        #     pro = [row[1] for row in dict_loc_dis.itertuples(index=False, name=False)]
        #     add_loc = []
        #     for i in range(self.comloc_num):
        #         while True:
        #             property = np.random.random()
        #             j = 0
        #             property_candidate = 0
        #             if property == 0:
        #                  j = 0
        #             else:
        #                 while property_candidate < property:  # 轮盘赌过程
        #                     property_candidate += pro[j]
        #                     j += 1
        #                 if property_candidate > property:
        #                     j -= 1
        #             if loc[j] not in add_loc:
        #                 add_loc.append(loc[j])
        #                 break
        #     community_unfreq_uncomlocs = add_loc
        # else:
        #     dict_loc_dis = self.distance_sort(core_user_centerpoint, [self.lat_lon[self.locids.index(loc)] for loc in list(set(self.golbal_freqlocs)-set(community_locs))], list(set(self.golbal_freqlocs)-set(community_locs)))
        #     dict_loc_dis = list(map(list, zip(*dict_loc_dis)))
        #     for i in range(len(dict_loc_dis)):
        #         dict_loc_dis[i][1] = math.exp(-dict_loc_dis[i][1])
        #     dict_loc_dis = pd.DataFrame(dict_loc_dis, columns=['locid', 'dis']).sort_values(by=['dis'],ascending=False).reset_index(drop=True)  # 按照距离升序排列
        #     sum = dict_loc_dis.dis.sum()
        #     dict_loc_dis.loc[:, 'dis'] = dict_loc_dis.loc[:, 'dis'] / sum
        #     loc = [row[0] for row in dict_loc_dis.itertuples(index=False, name=False)]
        #     pro = [row[1] for row in dict_loc_dis.itertuples(index=False, name=False)]
        #     add_loc = []
        #     for i in range(self.comloc_num - len(community_unfreq_uncomlocs)):
        #         while True:
        #             property = np.random.random()
        #             j = 0
        #             property_candidate = 0
        #             if property == 0:
        #                 j = 0
        #             else:
        #                 while property_candidate < property:  # 轮盘赌过程
        #                     property_candidate += pro[j]
        #                     j += 1
        #                 if property_candidate > property:
        #                     j -= 1
        #             if loc[j] not in add_loc:
        #                 add_loc.append(loc[j])
        #                 break
        #     community_unfreq_uncomlocs.extend(add_loc) **/
        # 选用全局位置/**
        # union_locs = set(community_unfreq_uncomlocs).intersection(set(self.golbal_freqlocs))   # 社区中非频繁费共同访问位置中的全局频繁访问位置
        # not_community_locids = set(self.golbal_freqlocs)-set(community_locs)
        # dict_loc_dis = self.distance_sort(core_user_centerpoint, [self.lat_lon[self.locids.index(loc)] for loc in list(not_community_locids.union(union_locs))], list(not_community_locids.union(union_locs)))
        # dict_loc_dis = list(map(list, zip(*dict_loc_dis)))
        # dict_loc_dis = pd.DataFrame(dict_loc_dis, columns=['locid', 'dis']).sort_values(by=['dis'], ascending=True).reset_index(drop=True)  # 按照距离升序排列
        # dict_loc_dis = dict_loc_dis[0:(self.comloc_num*2)]
        # community_unfreq_uncomlocs = np.random.choice(list(dict_loc_dis.locid.unique()), self.comloc_num, replace=False)**/
        for u in uids:
            # 需要扰动共同访问位置，保持频繁访问位置
            u_checkins = deepcopy(checkins[checkins.uid == u])
            is_core = u_checkins.is_core.values[0]
            u_freq_uncomloc = u_freq_uncomlocs[uids.index(u)]  # 如果没有可以选择的点，则用频繁访问位置中的非共同访问位置
            u_disturb_loc = u_disturb_locs[uids.index(u)]
            u_unfreq_uncomloc = u_unfreq_uncomloc_list[uids.index(u)]
            u_choose_loc = u_freq_uncomloc
            if len(u_choose_loc) == 0:
                u_choose_loc = user_k_freqloc[user_k_freqloc.uid == u].k_freqlocs.values[0]
            # u_choose_locs_select = []
            # loc_freq_nums = [[loc, len(u_checkins[u_checkins.locid == loc])] for loc in u_choose_loc]  # 记录用户的频繁访问位置的分布
            # sum = np.sum([x[1] for x in loc_freq_nums])
            # loc_freq_nums = [[line[0], float(line[1] / sum)] for line in loc_freq_nums]  # 位置、频率
            # disPr = []
            # candidate_loc = []
            # for loc in u_disturb_loc:
            #     loc_dis = self.distance_sort(self.lat_lon[self.locids.index(loc)], [self.lat_lon[self.locids.index(locid)] for locid in u_choose_loc], u_choose_loc)
            #     loc_dis = list(map(list, zip(*loc_dis)))
            #     loc_dis = sorted(loc_dis, key=lambda line: line[1], reverse=False)  # 按照距离升序排列
            #     sum = np.sum([line[1] for line in loc_dis])
            #     disPr = [float(line[1] / sum) for line in loc_dis]
            #     candidate_loc = [row[0] for row in loc_dis]
                # u_choose_locs_select.append([row[0] for row in loc_dis])
            # if len(u_choose_loc) >= 3:
            #     u_choose_locs_select = [row[0:2] for row in u_choose_locs_select]
            # elif len(u_choose_loc) <= 2:
            #     u_choose_locs_select = [row[0:1] for row in u_choose_locs_select]
            u_checkins1 = deepcopy(u_checkins[~u_checkins.locid.isin(u_disturb_loc)])   # 替换过后的用户签到记录
            for row in u_checkins.itertuples(index=False, name=False):   # 对需要扰动的共同访问位置进行扰动
                if row[4] in u_disturb_loc:
                    # 最近修改
                    # loc_dis = self.distance_sort(self.lat_lon[self.locids.index(row[4])], [self.lat_lon[self.locids.index(locid)] for locid in u_choose_loc], u_choose_loc)
                    # loc_dis = list(map(list, zip(*loc_dis)))
                    # for i in range(len(loc_dis)):
                    #     loc_dis[i][1] = math.exp(-loc_dis[i][1])
                    # loc_dis = pd.DataFrame(loc_dis, columns=['locid', 'dis']).sort_values(by=['dis'], ascending=False).reset_index(drop=True)  # 按照距离升序排列
                    # sum = loc_dis.dis.sum()
                    # loc_dis.loc[:, 'dis'] = loc_dis.loc[:, 'dis'] / sum
                    # disPr = [row[1] for row in loc_dis.itertuples(index=False, name=False)]
                    # candidate_loc = [row[0] for row in loc_dis.itertuples(index=False, name=False)]
                    # property = np.random.random()
                    # i = 0
                    # property_candidate = 0
                    # if property == 0:
                    #     i = 0
                    # else:
                    #     while property_candidate < property:  # 轮盘赌过程
                    #         property_candidate += disPr[i]
                    #         i += 1
                    #     if property_candidate > property:
                    #         i -= 1
                    # locid = candidate_loc[i]
                    # locid = np.random.choice(u_choose_locs_select[u_disturb_loc.index(row[4])])
                    locid = np.random.choice(u_choose_loc)
                    u_lat_lon = self.lat_lon[self.locids.index(locid)]
                    lat = u_lat_lon[0]
                    lng = u_lat_lon[1]
                    grid_id = self.grid_id[self.locids.index(locid)]
                    if self.is_replace_locid(u_checkins1, row[8], u_lat_lon):
                        ano_checkin = [row[0], row[1], lat, lng, locid, grid_id, row[6], row[7], row[8], row[5], row[4]]
                        ano_checkins.append(ano_checkin)
                        add_checkins_pd = pd.DataFrame([ano_checkin], columns=['uid', 'time', 'latitude', 'longitude', 'locid', 'grid_id', 'clusterid', 'is_core', 'timestamp', 'grid_id_before','locid_before'])
                        u_checkins1 = pd.concat([u_checkins1, add_checkins_pd], ignore_index=True)
                        u_checkins1 = u_checkins1.reset_index(drop=True)
                else:
                    ano_checkins.append(list(row))
            # 如果是社区内不需要发布的特征已经满足了
            # 如果社区内用户的频繁访问位置没有3个，为了在后期不增加频繁访问位置的相似性，需要增加不同的频繁访问位置
            u_freqloc = len(user_k_freqloc[user_k_freqloc.uid == u].k_freqlocs.values[0])
            u_freqlocids = user_k_freqloc[user_k_freqloc.uid == u].k_freqlocs.values[0]     # 如果频繁访问位置的个数小于1,为了保证顺序，需要增加一条记录
            u_frelocids_nums = user_k_freqloc[user_k_freqloc.uid == u].freqloc_nums.values[0]
            temp = {"locid": u_freqlocids, "cnt": u_frelocids_nums}
            u_freloc_lack = pd.DataFrame(temp)
            u_freloc_lack = u_freloc_lack[u_freloc_lack.cnt == 1].locid.values
            for locid in u_freloc_lack:
                checkin = deepcopy(self.checkins[self.checkins.locid == locid]).reset_index(drop=True)
                add_checkin = deepcopy(checkin[0:1].values.tolist()[0])
                add_checkin[0] = u
                add_checkin[5] = -1
                add_checkin[6] = clusterId
                add_checkin[7] = is_core
                add_checkin[9] = -1
                ano_checkins.append(add_checkin)
                add_checkins_pd = pd.DataFrame([add_checkin], columns=['uid', 'time', 'latitude', 'longitude', 'locid', 'grid_id', 'clusterid', 'is_core', 'timestamp', 'grid_id_before', 'locid_before'])
                u_checkins1 = pd.concat([u_checkins1, add_checkins_pd], ignore_index=True)
                u_checkins1 = u_checkins1.reset_index(drop=True)
            while u_freqloc < k:
                while True:
                    locid = np.random.choice(self.golbal_freqlocs)
                    if (locid not in community_unfreq_uncomlocs) & (locid not in list(u_checkins1.locid.unique())):
                    # if (locid not in community_locs) & (locid not in list(u_checkins1.locid.unique())):
                        break
                checkin = deepcopy(self.checkins[self.checkins.locid == locid]).reset_index(drop=True)
                timestamp_list = list(checkin.timestamp.values)  # 时间戳和记录是对应的
                num = 0
                for i in range(2):
                    index = self.reach_point(deepcopy(u_checkins1), timestamp_list, self.lat_lon[self.locids.index(locid)])
                    if index == -1:
                        break
                    else:
                        add_checkin = deepcopy(checkin[index:(index + 1)].values.tolist()[0])
                        add_checkin[0] = u
                        add_checkin[5] = -1
                        add_checkin[6] = clusterId
                        add_checkin[7] = is_core
                        add_checkin[9] = -1
                        ano_checkins.append(add_checkin)
                        add_checkins_pd = pd.DataFrame([add_checkin], columns=['uid', 'time', 'latitude', 'longitude', 'locid', 'grid_id', 'clusterid', 'is_core','timestamp', 'grid_id_before', 'locid_before'])
                        u_checkins1 = pd.concat([u_checkins1, add_checkins_pd], ignore_index=True)
                        u_checkins1 = u_checkins1.reset_index(drop=True)
                    num += 1
                if num != 0:
                    u_freqloc += 1
            # 为了提高社区内不需要发布的特征的相似度，需要增加签到记录
            u_add_locs = list(set(community_unfreq_uncomlocs) - set(u_unfreq_uncomloc))
            # u_add_locs = list(set(community_freq_uncomloc) - set(u_checkins.locid.unique()))
            for locid in u_add_locs:
                checkin = deepcopy(self.checkins[self.checkins.locid == locid]).reset_index(drop=True)
                timestamp_list = list(checkin.timestamp.values)   # 时间戳和记录是对应的
                index = self.reach_point(deepcopy(u_checkins1), timestamp_list, self.lat_lon[self.locids.index(locid)])   # 用户判断该点是否可以作为可选点
                if index == -1:
                    # 最新修改 /**
                    # add_checkins_pd = deepcopy(checkin[0:1])
                    # add_checkins_pd.loc[0, 'uid'] = u
                    # add_checkins_pd.loc[0, 'grid_id'] = -1  # 将新增的位置的grid_id和 grid_id_after赋值为-1,表示这条记录是新增的
                    # add_checkins_pd.loc[0, 'clusterid'] = clusterId
                    # add_checkins_pd.loc[0, 'is_core'] = is_core
                    # add_checkins_pd.loc[0, 'grid_id_before'] = -1 **/
                    continue
                else:
                    add_checkin = deepcopy(checkin[index:(index+1)].values.tolist()[0])
                    add_checkin[0] = u
                    add_checkin[5] = -1    # 将新增的位置的grid_id和 grid_id_after赋值为-1,表示这条记录是新增的
                    add_checkin[6] = clusterId
                    add_checkin[7] = is_core
                    add_checkin[9] = -1
                    ano_checkins.append(add_checkin)
                    add_checkins_pd = pd.DataFrame([add_checkin], columns=['uid', 'time', 'latitude', 'longitude', 'locid', 'grid_id', 'clusterid', 'is_core', 'timestamp', 'grid_id_before', 'locid_before'])
                u_checkins1 = pd.concat([u_checkins1, add_checkins_pd], ignore_index=True)
                u_checkins1 = u_checkins1.reset_index(drop=True)
        # 最新修改/**
        # while True:
        #     user = np.random.randint(7000000, 8000000)
        #     if user not in self.add_dummy_user:
        #         self.add_dummy_user.append(user)
        #         break
        # for locid in community_unfreq_uncomlocs:          # 为虚假用户增加共同访问位置提高扰动特征的相似度
        #     checkin = deepcopy(self.checkins[self.checkins.locid == locid]).reset_index(drop=True)
        #     add_checkin = deepcopy(checkin.sample(n=1).values.tolist()[0])
        #     add_checkin[0] = user
        #     add_checkin[5] = -1  # 将新增的位置的grid_id和 grid_id_after赋值为-1,表示这条记录是新增的
        #     add_checkin[6] = clusterId
        #     add_checkin[7] = 0
        #     add_checkin[9] = -1
        #     ano_checkins.append(add_checkin)
        # ano_checkins = pd.DataFrame(ano_checkins, columns=['uid', 'time', 'latitude', 'longitude', 'locid', 'grid_id', 'clusterid', 'is_core', 'timestamp', 'grid_id_before', 'locid_before'])
        # unvisitedCluster_checkins = self.checkins[~self.checkins.clusterid.isin(self.visitedCluster)].reset_index(drop=True)
        # if len(self.visitedCluster) == 1:
        #     save_ano_checkins = pd.concat([ano_checkins, unvisitedCluster_checkins], ignore_index=True).reset_index(drop=True)
        # else:
        #     save_ano_checkins = pd.read_csv(self.path + "result_data/" + self.ano_checkins_tablename + "/" + self.city + "_70_" + str(self.user_num) + "_" + self.ano_checkins_tablename + ".csv", index_col=None, sep='\t',names=['uid', 'time', 'latitude', 'longitude', 'locid', 'grid_id', 'clusterid', 'is_core', 'timestamp', 'grid_id_before', 'locid_before'])
        #     save_ano_checkins = pd.concat([save_ano_checkins, ano_checkins, unvisitedCluster_checkins], ignore_index=True).reset_index(drop=True)
        # temp_checkins = pd.DataFrame(save_ano_checkins['locid'].value_counts()).reset_index()  # 获得所有位置中的频繁访问位置
        # temp_checkins.columns = ["locid", "cnt"]
        # temp_checkins = temp_checkins.sort_values(by=['cnt', 'locid'], ascending=False).reset_index(drop=True)
        # curr_globle_freqloc = set((temp_checkins[0:100].locid.unique()))
        # diff_freqloc = list(set(self.golbal_freqlocs) - curr_globle_freqloc)  # 扰动之后不在前100的频繁访问位置，需要使用虚假用户来调整全局频繁访问位置
        # repalce_freqloc = list(curr_globle_freqloc - (set(self.golbal_freqlocs)))  # 扰动之后在前100的位置
        # repalce_freqloc = deepcopy(temp_checkins[temp_checkins.locid.isin(repalce_freqloc)]).sort_values(by=['cnt'], ascending=False).reset_index(drop=True)
        # # 还要保证顺序
        # if len(diff_freqloc) > 1:
        #     second_cnt = temp_checkins[temp_checkins.locid == repalce_freqloc[1:2].locid.values[0]].cnt.values[0]  # 第二个位置
        #     index = temp_checkins[temp_checkins.locid == repalce_freqloc[1:2].locid.values[0]].index.tolist()[0]   # 第二个位置位置的索引
        #     temp_freqlocs = deepcopy(temp_checkins[index:100])
        #     repalce_freqloc.drop(0, inplace=True)
        #     for i in range(len(repalce_freqloc)):
        #         temp_freqlocs.drop(temp_checkins[temp_checkins.locid == repalce_freqloc[i:(i+1)].locid.values[0]].index.tolist()[0], inplace=True)
        #     differ_freqlocs = deepcopy(temp_checkins[temp_checkins.locid.isin(diff_freqloc)])
        #     temp_freqlocs = pd.concat([temp_freqlocs, differ_freqlocs], ignore_index=True).reset_index(drop=True)
        #     for i in range(len(temp_freqlocs)):
        #         lack_cnt = second_cnt - temp_freqlocs[i:(i+1)].cnt.values[0] + 1
        #         diffloc_checkins = deepcopy(self.checkins[self.checkins.locid == temp_freqlocs[i:(i+1)].locid.values[0]])
        #         diffloc_checkins = diffloc_checkins.sample(n=lack_cnt, replace=True)
        #         diffloc_checkins.loc[:, 'uid'] = user
        #         diffloc_checkins.loc[:, 'grid_id'] = -1
        #         diffloc_checkins.loc[:, 'clusterid'] = clusterId
        #         diffloc_checkins.loc[:, 'is_core'] = 0
        #         diffloc_checkins.loc[:, 'grid_id_before'] = -1
        #         ano_checkins = pd.concat([ano_checkins, diffloc_checkins], ignore_index=True)
        #     ano_checkins = ano_checkins.reset_index(drop=True)**/
        self.save_ano_checkins(ano_checkins)  # 扰动后的匿名数据

    def save_ano_checkins(self, ano_checkin):
        ano_checkins = pd.DataFrame(ano_checkin)
        ano_checkins.to_csv(self.path + "result_data/"+self.ano_checkins_tablename + "/" + self.city + "_70_" + str(self.user_num) + "_" + self.ano_checkins_tablename + ".csv", header=None, index=None, sep='\t', mode='a')

    def comnunity_disturb(self, checkins, k):
        temp_checkins = deepcopy(checkins)
        temp_checkins = pd.DataFrame(temp_checkins['locid'].value_counts()).reset_index()  # 获得所有位置中的频繁访问位置
        temp_checkins.columns = ["locid", "cnt"]
        temp_checkins = temp_checkins.sort_values(by=['cnt', 'locid'], ascending=False).reset_index(drop=True)
        temp_checkins = temp_checkins[0:100]
        self.golbal_freqlocs = list(temp_checkins.locid.unique())
        del temp_checkins
        gc.collect()

        checkins.loc[:, 'timestamp'] = checkins['time'].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))  # 将签到记录中的日期时间转换成时间戳
        checkins.loc[:, 'grid_id_before'] = checkins['grid_id']  # 重新复制一列
        checkins.loc[:, 'locid_before'] = checkins['locid']
        # 位置和经纬度的对应关系
        checkin1 = checkins.groupby(by=['locid', 'latitude', 'longitude', 'grid_id']).size().reset_index(name="locid_time")
        self.locids = [row[0] for row in checkin1.itertuples(index=False, name=False)]  # 记录locid对应的经纬度，以便在替换locid时将相应的位置数据也进行替换
        self.lat_lon = [[row[1], row[2]] for row in checkin1.itertuples(index=False, name=False)]
        self.grid_id = [row[3] for row in checkin1.itertuples(index=False, name=False)]
        del checkin1  # 释放checkin1的内存空间
        gc.collect()
        self.checkins = checkins

        community_checkins = checkins.groupby(["clusterid"])
        print(len(community_checkins))
        checkin_cnt = 0
        checkin_chunk_size = math.ceil(len(community_checkins) / 10)
        for group in community_checkins:
            if checkin_cnt % checkin_chunk_size == 0:  # finished the anonymization of a chunk of checkins打印一部分匿名化的结果
                print('%-3d%% work complete.' % (int(checkin_cnt / checkin_chunk_size) * 10))
            self.unsim_comloc_disturb(group[1], k)
            checkin_cnt += 1

        ano_checkins = pd.read_csv(self.path + "result_data/" + self.ano_checkins_tablename + "/" + self.city + "_70_" +str(self.user_num) + "_" + self.ano_checkins_tablename + ".csv", index_col=None,
                                   sep='\t', names=['uid', 'time', 'latitude', 'longitude', 'locid', 'grid_id', 'clusterid', 'is_core', 'timestamp', 'grid_id_before', 'locid_before'])
        ano_checkins = ano_checkins.ix[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]]
        ano_checkins.to_csv(self.path + "result_data/" + self.ano_checkins_tablename + "/" + self.city + "_70_" + str(self.user_num) + "_" + self.ano_checkins_tablename + ".csv", header=None, index=None, sep='\t')
        return ano_checkins


# if __name__ == "__main__":
#
#     # for k in [4, 5, 6, 7, 8, 9, 10]:
#     for k in [3]:
#         start = time.time()
#         pc = disturb_comloc(int(k), "comloc")
#         checkins = pd.read_csv("G:/pyfile/relation_protect/src/data/city_data/1.csv", delimiter="\t", index_col=None)
#         checkins = deepcopy(checkins[checkins.uid.isin([16509, 116410, 119171])])
#         print(checkins)
#         # checkins = pd.read_csv(
#         #     "G:/pyfile/relation_protect/src/data/result_data/1_comloc_" + str(k) + "_user_simple_community.data",
#         #     delimiter="\t", names=["uid", "time", "latitude", "longitude", "locid", "clusterid"], header=None)
#         # pc.comnunity_disturb(checkins, 3, 0.6)
#         checkins.loc[:, 'clusterid'] = -1
#         pc.unsim_comloc_disturb(checkins, 4)
#         end = time.time()
#         print("花费时间：", str(end-start))
#         # data = pd.DataFrame(data, columns=["uid", "time", "lat", "lng", "locid"])
#         # print(data)
#         # data.to_csv("G:/pyfile/relation_protect/src/data/city_data/test1.csv", sep='\t', index=False, header=False)
