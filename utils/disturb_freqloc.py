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
import decimal


class disturb_freqloc:

    def __init__(self, user_num, tablename, path, city, m, n, q, clusterlist, is_resemble, p_num_list, latInterval, lngInterval, lons_per_km, lats_per_km):
        self.user_num = user_num
        self.ano_checkins_tablename = tablename
        self.golbal_freqlocs = []
        self.lons_per_km = decimal.Decimal.from_float(lons_per_km)  # delta longitudes per kilo meter
        self.lats_per_km = decimal.Decimal.from_float(lats_per_km)  # delta latitudes per kilo meter
        self.m = m
        self.n = n
        self.path = path
        self.circle = self.circle_distance(q)
        self.clusterlist = clusterlist
        self.is_resemble = is_resemble
        self.p_num_list = p_num_list
        self.latInterval = latInterval
        self.lngInterval = lngInterval
        self.city = city

    def k_freq_loc(self, u, u_checkins, k):
        u_loc = pd.DataFrame(u_checkins['locid'].value_counts()).reset_index()  # 统计locid的不同值及其个数
        u_loc.columns = ["locid", "cnt"]
        u_loc = u_loc.sort_values(by=['cnt', 'locid'], ascending=False).reset_index(drop=True)
        if k <= len(u_loc):
            u_loc = u_loc[0:k]
        return [u, list(u_loc.locid.values)]

    def u_comloc(self, u1_checkins, u2_checkins, u1, u2):
        u1_locids = u1_checkins.locid.unique()
        u2_locids = u2_checkins.locid.unique()
        comloc = list(set(u1_locids).intersection(set(u2_locids)))
        return [u1, u2, len(comloc), comloc]

    # 两点之间的欧氏距离计算
    def euclidean_distance(self, loc1, loc2):
        return math.sqrt(((loc1[1] - loc2[1]) / float(self.lons_per_km)) ** 2 + ((loc1[0] - loc2[0]) / float(self.lats_per_km)) ** 2)

    def circle_distance(self, q):
        return -math.log(q)

    def distance(self, grid1, grid2):
        index_i1, index_i2 = int(grid1 / self.m), int(grid2 / self.m)
        index_j1, index_j2 = grid1 % self.m, grid2 % self.m
        i_interval = abs(index_i1 - index_i2)
        j_interval = abs(index_j1 - index_j2)
        return math.sqrt((i_interval * self.latInterval / self.lats_per_km) ** 2 + (j_interval * self.lngInterval / self.lons_per_km) ** 2)

    def choose_locids(self, locids, nums, community_locids, grid_ids):  # 选取满足条件的locid
        """
        :param locids:   核心用户的频繁访问位置
        :param nums:
        :param community_locids:  社区内已有的位置
        :param grid_ids:          核心用户的的频繁访问位置所在的网格
        :return:        返回与核心用户的频繁访问位置满足相似度条件的“邻近位置”
        """
        num = 0
        statisfied_locids = []
        i = 0
        for locid in locids:
            repalce_grid_ids = []
            left = grid_ids[i] - 1
            right = grid_ids[i] + 1
            top = grid_ids[i] - self.m
            below = grid_ids[i] + self.m
            if (left != -1) & (left % self.m != self.m - 1):  # 判断当前位置的上下左右是否可以访问
                distance = self.distance(grid_ids[i], left)
                if distance <= self.circle:
                    repalce_grid_ids.append(left)
            if (right != self.m * self.n) & (right % self.m != 0):
                distance = self.distance(grid_ids[i], right)
                if distance <= self.circle:
                    repalce_grid_ids.append(right)
            if top > 0:
                distance = self.distance(grid_ids[i], top)
                if distance <= self.circle:
                    repalce_grid_ids.append(top)
            if below < self.m * self.n:
                distance = self.distance(grid_ids[i], below)
                if distance <= self.circle:
                    repalce_grid_ids.append(below)
            repalce_grid_ids.append(grid_ids[i])
            close_locid = self.checkins[self.checkins.grid_id.isin(repalce_grid_ids)].locid.unique()
            close_locid = list(set(close_locid)-set(community_locids)-set(statisfied_locids)-set(locids))  # 防止选到相同的位置
            lat_lon = self.lat_lon[self.locids.index(locid)]
            for c_locid in close_locid:
                lat_lon_close = self.lat_lon[self.locids.index(c_locid)]
                distance = self.euclidean_distance(lat_lon, lat_lon_close)
                if distance != 0:
                # if (distance <= self.circle) & (distance != 0):
                    statisfied_locids.append(c_locid)
                    num += 1
                if num == nums:
                    return statisfied_locids
            i += 1
        return statisfied_locids

    def recursion_user(self, resemble_list, user):
        temp = list(resemble_list[resemble_list.start == user].end.values)
        if len(temp) == 0:
            return temp
        else:
            temp1 = deepcopy(temp)
            for u in temp1:
                list_temp = self.recursion_user(resemble_list, u)
                temp.extend(list_temp)
        return temp

    def unsim_comloc_disturb(self, checkins, k):  # k为用户的前k个频繁访问位置、m为前m%的共同访问位置，checkins是签到数据
        uids = checkins.uid.unique()
        core_user = checkins[checkins.is_core == 1].uid.values[0]  # 社区内的核心用户
        clusterId = checkins.clusterid.values[0]  # 社区id
        p_nums = self.p_num_list[self.p_num_list.clusterId == clusterId].p_nums.values[0]
        core_num = mp.cpu_count()
        checkins = checkins.sort_values(by=['uid'], ascending=True).reset_index(drop=True)  # uid的降序排列
        user_k_freqloc = Parallel(n_jobs=core_num)(delayed(self.k_freq_loc)(u, checkins[checkins.uid == u], k) for u in uids)
        user_k_freqloc = pd.DataFrame(user_k_freqloc, columns=['uid', 'k_freqlocs'])
        pairs = pd.DataFrame(list(combinations(uids, 2)), columns=['u1', 'u2'])
        pairs_comloc = Parallel(n_jobs=core_num)(delayed(self.u_comloc)(checkins[checkins.uid == row[0]], checkins[checkins.uid == row[1]], row[0], row[1]) for row in pairs.itertuples(index=False, name=False))
        pairs_comloc = pd.DataFrame(pairs_comloc, columns=["u1", "u2", "comloc_num", "comlocs"])
        pairs_comloc = pairs_comloc.sort_values(by='comloc_num', ascending=False).reset_index(drop=True)
        uids = list(uids)
        u_disturb_locs = []   # 用于记录每个用户的需要扰动的位置
        u_comlocs = []        # 用于记录每个用户的共同访问位置
        for u in uids:
            comlocs = pairs_comloc[(pairs_comloc.u1 == u) | (pairs_comloc.u2 == u)].comlocs
            u_comloc = []                         # 用户的共同访问位置
            list(map(lambda x: u_comloc.extend(x), comlocs))
            u_comloc = list(set(u_comloc))
            u_comlocs.append(u_comloc)
            u_freqloc = user_k_freqloc[user_k_freqloc.uid == u].k_freqlocs.values[0]    # 用户的频繁访问位置
            u_disturb_locs.append(list(set(u_freqloc) - set(u_comloc)))    # 频繁位置中的非共同访问访问位置，即需要扰动的共同访问位置

        resemble_list = self.is_resemble[self.clusterlist.index(clusterId)]  # 获得社区内的相似用户对
        resemble_list = pd.DataFrame(resemble_list, columns=['start', 'end'])
        simusers = []
        temp_uids = deepcopy(uids)
        for i in range(p_nums):
            temp_core_user = temp_uids[0]
            core_simusers = [temp_core_user]      # 将每个具有不同特征的用户放在第一个
            core_simusers.extend(self.recursion_user(resemble_list, temp_core_user))  # 跟核心用户相似的用户， 先对与核心用户相似的用户进行位置扰动
            simusers.append(core_simusers)
            for user in core_simusers:
                temp_uids.remove(user)
        ano_checkins = pd.DataFrame()
        # 核心用户先进行位置选择,这样便于选择频繁访问位置
        core_user_checkins = deepcopy(checkins[checkins.uid == core_user]).reset_index(drop=True)
        core_locids = set(core_user_checkins.locid.unique())
        core_user_disturb_locids = u_disturb_locs[uids.index(core_user)]
        core_user_choose_locids = list(core_locids-set(core_user_disturb_locids))  # 除频繁非共同访问位置外的所有位置
        community_locids = list(checkins.locid.unique())
        satisfied_locids = []
        temp_list = []
        i = 0
        if len(core_user_disturb_locids) == 0:  # 核心用户没有扰动位置,则频繁访问位置为自己的频繁访问位置，如果不够则更换位置
            core_user_choose_locids = user_k_freqloc[user_k_freqloc.uid == core_user].k_freqlocs.values[0]
            core_grid_ids = []
            for loc in core_user_choose_locids:
                core_grid_ids.append(self.grid_id[self.locids.index(loc)])
            satisfied_locids = self.choose_locids(core_user_choose_locids, p_nums - 1, community_locids, core_grid_ids)
            if len(satisfied_locids) < p_nums - 1:
                change_locid = core_user_choose_locids[len(core_user_choose_locids)-1]
                num = 0
                while len(satisfied_locids) < p_nums - 1:
                    num += 1
                    del core_user_choose_locids[len(core_user_choose_locids) - 1]  # 删除最后一个元素
                    del core_grid_ids[len(core_grid_ids) - 1]
                    locid = np.random.choice(list((set(self.golbal_freqlocs) - set(community_locids)) - set(core_user_choose_locids)))
                    if num > 20:
                        locid = np.random.choice(list((set(self.locids) - set(community_locids)) - set(core_user_choose_locids)))
                    core_user_choose_locids.append(locid)
                    core_grid_ids.append(self.grid_id[self.locids.index(locid)])
                    satisfied_locids = self.choose_locids(core_user_choose_locids, p_nums - 1, community_locids, core_grid_ids)
                locid = core_user_choose_locids[len(core_user_choose_locids)-1]
                lat_lon = self.lat_lon[self.locids.index(locid)]
                grid_id = self.grid_id[self.locids.index(locid)]
                ano_checkin = deepcopy(core_user_checkins[core_user_checkins.locid == change_locid]).reset_index(drop=True)  # 为每个频繁位置添加一条记录，防止排序时出现不稳定的情况
                ano_checkin = deepcopy(ano_checkin[0:1])
                core_user_checkins.loc[core_user_checkins.locid == change_locid, 'latitude'] = lat_lon[0]
                core_user_checkins.loc[core_user_checkins.locid == change_locid, 'longitude'] = lat_lon[1]
                core_user_checkins.loc[core_user_checkins.locid == change_locid, 'grid_id'] = grid_id
                core_user_checkins.loc[core_user_checkins.locid == change_locid, 'locid'] = locid
                ano_checkin1 = deepcopy(core_user_checkins[0:1])  # 为每个频繁位置添加一条记录，防止排序时出现不稳定的情况
                ano_checkin1.loc[:, "latitude"] = lat_lon[0]
                ano_checkin1.loc[:, "longitude"] = lat_lon[1]
                ano_checkin1.loc[:, "grid_id"] = grid_id
                ano_checkin1.loc[:, "locid"] = locid
                core_user_checkins = pd.concat([core_user_checkins, ano_checkin, ano_checkin1], ignore_index=True).reset_index(drop=True)
            ano_checkins = pd.concat([ano_checkins, core_user_checkins], ignore_index=True).reset_index(drop=True)
        core_user_remain_freqloc = list(set(user_k_freqloc[user_k_freqloc.uid == core_user].k_freqlocs.values[0])-set(core_user_disturb_locids))
        core_user_choose_locids = list(set(core_user_choose_locids) - set(core_user_remain_freqloc))
        while len(satisfied_locids) < p_nums - 1:
            if len(satisfied_locids) != 0:
                satisfied_locids.clear()
            core_user_choose_locids = list(set(core_user_choose_locids) - set(temp_list))
            temp_list = []
            while len(core_user_choose_locids) < len(core_user_disturb_locids):  # 替换需要扰动的位置过少，只能从全局进行选择
                i += 1
                locid = np.random.choice(list((set(self.golbal_freqlocs) - set(community_locids)) - set(core_user_choose_locids)))
                if i >= 20:      # 循环次数太多，没有可以选的点,选择更换频繁访问位置
                    locid = np.random.choice(list((set(self.locids) - set(community_locids)) - set(core_user_choose_locids)))
                if (locid not in community_locids) & (locid not in core_user_choose_locids):  # 不增加新的共同访问位置
                    core_user_choose_locids.append(locid)
                    temp_list.append(locid)
            core_grid_ids = []
            if (len(core_user_choose_locids) > len(core_user_disturb_locids)) & (len(core_user_disturb_locids) != 0):
                core_user_choose_locids = list(np.random.choice(core_user_choose_locids, len(core_user_disturb_locids), replace=False))
            for loc in core_user_choose_locids:
                core_grid_ids.append(self.grid_id[self.locids.index(loc)])
            core_user_freloc_new = deepcopy(core_user_choose_locids)
            for loc in core_user_remain_freqloc:     # 将不需要扰动的频繁访问位置添加进来，寻找邻近位置
                core_user_freloc_new.append(loc)
                core_grid_ids.append(self.grid_id[self.locids.index(loc)])
            satisfied_locids = self.choose_locids(core_user_freloc_new, p_nums-1, community_locids, core_grid_ids)
        if len(core_user_disturb_locids) != 0:
            for i in range(len(core_user_disturb_locids)):
                locid = core_user_choose_locids[i]
                lat_lon = self.lat_lon[self.locids.index(locid)]
                grid_id = self.grid_id[self.locids.index(locid)]
                ano_checkin = deepcopy(core_user_checkins[0:1])  # 为每个频繁位置添加一条记录，防止排序时出现不稳定的情况
                ano_checkin.loc[:, "latitude"] = lat_lon[0]
                ano_checkin.loc[:, "longitude"] = lat_lon[1]
                ano_checkin.loc[:, "grid_id"] = grid_id
                if isinstance(locid, str):
                    ano_checkin.loc[:, "locid"] = locid.encode('UTF-8', 'ignore').decode('UTF-8')
                else:
                    ano_checkin.loc[:, "locid"] = locid
                core_user_checkins.loc[core_user_checkins.locid == core_user_disturb_locids[i], 'latitude'] = lat_lon[0]
                core_user_checkins.loc[core_user_checkins.locid == core_user_disturb_locids[i], 'longitude'] = lat_lon[1]
                core_user_checkins.loc[core_user_checkins.locid == core_user_disturb_locids[i], 'grid_id'] = grid_id
                if isinstance(locid, str):
                    core_user_checkins.loc[core_user_checkins.locid == core_user_disturb_locids[i], 'locid'] = locid.encode('UTF-8', 'ignore').decode('UTF-8')
                else:
                    core_user_checkins.loc[core_user_checkins.locid == core_user_disturb_locids[i], 'locid'] = locid

                core_user_checkins = pd.concat([core_user_checkins, ano_checkin], ignore_index=True).reset_index(drop=True)  # 为什么没用啊
            ano_checkins = pd.concat([ano_checkins, core_user_checkins], ignore_index=True).reset_index(drop=True)

        core_user_loc = pd.DataFrame(core_user_checkins['locid'].value_counts()).reset_index()
        core_user_loc.columns = ["locid", "cnt"]
        core_user_loc = core_user_loc.sort_values(by=['cnt', 'locid'], ascending=False).reset_index(drop=True)
        core_user_freqlocs = core_user_loc[0:k].locid.values   # 获得核心用户的改变后的频繁访问位置,然后根据核心用户的频繁访问位置进行相似度的提高扩展
        # 将已经选择的用户放在一起
        all_choose_locids = []
        all_choose_locids.extend(core_user_freqlocs)
        satisfied_locids.insert(0, np.random.choice(core_user_freqlocs))  # 将跟核心用户相似的用户选择的位置放在第一个
        temp_uids = deepcopy(uids)
        temp_uids.remove(core_user)
        all_choose_locids.extend(satisfied_locids)
        all_choose_locids.extend(community_locids)
        all_choose_locids = list(set(all_choose_locids))
        groble_locids = list(set(self.golbal_freqlocs)-set(all_choose_locids))
        for u in temp_uids:
            for i in range(p_nums):
                if u in simusers[i]:
                    num = 0
                    sim_loc = satisfied_locids[i]
                    u_checkins = deepcopy(checkins[checkins.uid == u]).reset_index(drop=True)
                    u_locids = set(u_checkins.locid.unique())
                    u_disturb_loc = u_disturb_locs[uids.index(u)]
                    u_freqloc = user_k_freqloc[user_k_freqloc.uid == u].k_freqlocs.values[0]
                    u_choose_locids = list(u_locids - set(u_freqloc))
                    u_choose_locids.insert(0, sim_loc)
                    while len(u_disturb_loc) > len(u_choose_locids):
                        num += 1
                        locid = np.random.choice(groble_locids)
                        if num >= 20:   # 循环次数超过50,则从更广的范围找位置
                            locid = np.random.choice(list(set(self.locids)-set(all_choose_locids)))
                        if (locid not in all_choose_locids) & (locid not in u_choose_locids):
                            u_choose_locids.append(locid)
                    if (len(u_choose_locids) > len(u_disturb_loc)) & (len(u_disturb_loc) != 0):
                        u_choose_locids = u_choose_locids[0:len(u_disturb_loc)]
                    all_choose_locids.extend(u_choose_locids)
                    if len(u_disturb_loc) == 0:
                        locid = u_choose_locids[0]
                        change_locid = u_freqloc[len(u_freqloc)-1]
                        lat_lon = self.lat_lon[self.locids.index(locid)]
                        grid_id = self.grid_id[self.locids.index(locid)]
                        ano_checkin = deepcopy(u_checkins[u_checkins.locid == change_locid]).reset_index(drop=True)  # 为每个频繁位置添加一条记录，防止排序时出现不稳定的情况
                        ano_checkin = deepcopy(ano_checkin[0:1])
                        ano_checkin1 = deepcopy(u_checkins[0:1])  # 为每个频繁位置添加一条记录，防止排序时出现不稳定的情况
                        ano_checkin1.loc[:, "latitude"] = lat_lon[0]
                        ano_checkin1.loc[:, "longitude"] = lat_lon[1]
                        ano_checkin1.loc[:, "grid_id"] = grid_id
                        if isinstance(locid, str):
                            ano_checkin1.loc[:, "locid"] = locid.encode('UTF-8', 'ignore').decode('UTF-8')
                        else:
                            ano_checkin1.loc[:, "locid"] = locid
                        u_checkins.loc[u_checkins.locid == change_locid, 'latitude'] = lat_lon[0]
                        u_checkins.loc[u_checkins.locid == change_locid, 'longitude'] = lat_lon[1]
                        u_checkins.loc[u_checkins.locid == change_locid, 'grid_id'] = grid_id
                        u_checkins.loc[u_checkins.locid == change_locid, 'locid'] = locid.encode('UTF-8', 'ignore').decode('UTF-8')
                        u_checkins = pd.concat([u_checkins, ano_checkin, ano_checkin1], ignore_index=True).reset_index(drop=True)
                    else:
                        for i in range(len(u_disturb_loc)):   # 如果没有需要替换的位置，说明频繁访问位置都是共同访问位置，不需要做任何改动
                            locid = u_choose_locids[i]
                            lat_lon = self.lat_lon[self.locids.index(locid)]
                            grid_id = self.grid_id[self.locids.index(locid)]
                            ano_checkin = deepcopy(u_checkins[0:1])  # 为每个频繁位置添加一条记录，防止排序时出现不稳定的情况
                            ano_checkin.loc[:, "latitude"] = lat_lon[0]
                            ano_checkin.loc[:, "longitude"] = lat_lon[1]
                            ano_checkin.loc[:, "grid_id"] = grid_id
                            if isinstance(locid, str):
                                ano_checkin.loc[:, "locid"] = locid.encode('UTF-8', 'ignore').decode('UTF-8')
                            else:
                                ano_checkin.loc[:, "locid"] = locid
                            u_checkins.loc[u_checkins.locid == u_disturb_loc[i], 'latitude'] = lat_lon[0]
                            u_checkins.loc[u_checkins.locid == u_disturb_loc[i], 'longitude'] = lat_lon[1]
                            u_checkins.loc[u_checkins.locid == u_disturb_loc[i], 'grid_id'] = grid_id
                            u_checkins.loc[u_checkins.locid == u_disturb_loc[i], 'locid'] = locid.encode('UTF-8', 'ignore').decode('UTF-8')
                            u_checkins = pd.concat([u_checkins, ano_checkin], ignore_index=True).reset_index(drop=True)
                    ano_checkins = pd.concat([ano_checkins, u_checkins], ignore_index=True).reset_index(drop=True)
                    break
        self.save_ano_checkins(ano_checkins)  # 扰动后的匿名数据

    def save_ano_checkins(self, ano_checkin):
        ano_checkin.to_csv(self.path + "result_data/" + self.ano_checkins_tablename + "/" + self.city + "_1_" + str(self.user_num) + "_" + self.ano_checkins_tablename + ".csv", header=None, index=None, sep='\t', mode='a')

    def comnunity_disturb(self, checkins, k):
        temp_checkins = deepcopy(checkins)
        temp_checkins = pd.DataFrame(temp_checkins['locid'].value_counts()).reset_index()  # 获得所有位置中的频繁访问位置
        temp_checkins.columns = ["locid", "cnt"]
        temp_checkins = temp_checkins.sort_values(by=['cnt', 'locid'], ascending=False).reset_index(drop=True)
        temp_checkins = temp_checkins[100:300]
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
        # 将数据按照uid升序进行排列
        ano_checkins = pd.read_csv(self.path + "result_data/" + self.ano_checkins_tablename + "/" + self.city + "_1_" +
                                   str(self.user_num) + "_" + self.ano_checkins_tablename + ".csv", index_col=None,
                                   sep='\t', names=['uid', 'time', 'latitude', 'longitude', 'locid', 'grid_id', 'clusterid', 'is_core', 'timestamp', 'grid_id_before', 'locid_before'])
        # ano_checkins = ano_checkins.sort_values(by=['uid']).reset_index(drop=True)
        ano_checkins = ano_checkins.ix[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]]
        ano_checkins.to_csv(self.path+"result_data/" + self.ano_checkins_tablename + "/" + self.city + "_1_" + str(self.user_num) + "_" + self.ano_checkins_tablename + ".csv", header=None, index=None, sep='\t')
        return ano_checkins

# if __name__ == "__main__":
#
#     # for k in [4, 5, 6, 7, 8, 9, 10]:
#     for k in [3]:
#         start = time.time()
#         pc = disturb_freqloc(int(k), "comloc")
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