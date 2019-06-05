#!/usr/bin/env python
# encoding: utf-8

import time
import pandas as pd
from grid_divide import grid_divide
from copy import deepcopy
import math
from sys import argv

class commutity_security():

    def __init__(self, method, k, city, lats_per_km, lons_per_km):

        self.method = method
        self.lons_per_km = lons_per_km
        self.lats_per_km = lats_per_km
        self.city = city

    def set_checkins(self, checkins, checkins_obf):
        self.checkin = checkins
        self.clusternum = len(deepcopy(self.checkin.groupby(by=['clusterid'])))
        self.checkin_obf = checkins_obf
        uids = list(self.checkin.uid.unique())
        self.user_nums = len(uids)
        print(self.user_nums)
        self.checkin_obf = self.checkin_obf[self.checkin_obf.uid.isin(uids)]

    def cal_security(self, k):
        print(k)
        from security_unility import security_unility
        security_unility = security_unility(self.lons_per_km, self.lats_per_km)
        security_unility.set_checkins(self.checkin, self.checkin_obf, "G:/pyfile/relation_protect/src/data/" + "city_data/", self.city)
        b = security_unility.b_security(3)
        c = security_unility.c_security(100)
        d = security_unility.d_security()
        e = security_unility.e_security()
        f = security_unility.f_security()
        g = security_unility.g_security()
        a = security_unility.unility()
        file = open("G:/pyfile/relation_protect/src/data/"+"result_data/community_result.txt", 'a', encoding='UTF-8')
        # file.write(self.method+':'+'可用性:' + str(k) + ' ' + 'b:' + str(b) + ' ' + 'c:' + str(c)
        #            + ' ' + 'd:' + str(d) + ' ' + 'e:' + str(e)
        #            + ' ' + 'f:' + str(f) + ' ' + 'g:' + str(g)
        #            + ' ' + 'user:' + str(self.user_nums) + ' ' + 'cluster_num:' + str(self.clusternum)
        #            + ' ' + '安全性：' + 'a:' + str(a) + ' 11'
        #            + '\n')
        file.write(self.method + ' ' + str(k) + ' ' + str(a)
                   + ' ' + str(b) + ' ' + str(c)
                   + ' ' + str(d) + ' ' + str(e)
                   + ' ' + str(f) + ' ' + str(g)
                   + ' ' + str(self.user_nums) + ' ' + str(self.clusternum)
                   + '\n')
        # file.write(self.method + ':' + '安全性:' + str(k) + ' ' + 'b:' + str(b) + ' ' + 'c:' + str(c) + '\n')
        # a = security.all_a_security1()
        # file.write(self.method + ':' + 'c:' + str(k) + ' ' + str(c) + '\n')
        # file.write(self.method + ':' + 'g:' + str(k) + ' ' + str(g) + '\n')
        # file.write(self.method + ':' + 'a:' + str(k) + ' ' + str(a) + '\n')
        file.close()

    def cal_utility(self, a, k):
        print()
        print(k)
        from utility import utility
        utility = utility()
        checkin = self.checkin.ix[:, [0, 1, 2, 3, 4]]
        utility.set_checkins(checkin, self.checkin_obf, "G:/pyfile/relation_protect/src/data/city_data/", "G:/pyfile/relation_protect/src/data/result_data/" + self.method, "1", str(k) + "_community",self.method)
        # utility.checkin_sim_list2()
        k_utility = utility.sim_utility2()
        file = open("G:/pyfile/relation_protect/src/data/result_data/community_result.txt", 'a', encoding='UTF-8')
        file.write(self.method+':'+'可用性:' + str(k) + ' ' + str(k_utility) + '\n')
        file.close()


if __name__ == "__main__":
    start = time.time()
    k=3
    # for k in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    # for k in [3, 4, 5]:
    #     checkin = pd.read_csv("G:/pyfile/relation_protect/src/data/result_data/freqloc/SNAP_NY_1_4_" + str(k) + "_freqloc.csv", delimiter="\t",  names=['uid', 'time', 'latitude', 'longitude', 'locid', 'grid_id', 'clusterid', 'is_core', 'grid_id_before', 'locid_before'], header=None)
    #     checkin_obf = pd.read_csv("G:/pyfile/relation_protect/src/data/" + "city_data/SNAP_NY_1.csv", delimiter="\t", index_col=None)
    # checkin = pd.read_csv("G:/pyfile/relation_protect/src/data/result_data/freqloc/SF_14_" + str(k) + "_freqloc.csv", delimiter="\t", names=['uid', 'time', 'latitude', 'longitude', 'locid', 'grid_id', 'clusterid', 'is_core', 'grid_id_before', 'locid_before'], header=None)
    # checkin_obf = pd.read_csv("G:/pyfile/relation_protect/src/data/" + "city_data/SF.csv", delimiter="\t", index_col=None)
    # checkin = pd.read_csv("G:/pyfile/relation_protect/src/data/result_data/freqloc/FS_NY_1_1_" + str(k) + "_freqloc.csv",
    #                   delimiter="\t",
    #                   names=['uid', 'time', 'latitude', 'longitude', 'locid', 'grid_id', 'clusterid', 'is_core',
    #                          'grid_id_before', 'locid_before'], header=None)
    checkin_obf = pd.read_csv("G:/pyfile/relation_protect/src/data/" + "city_data/1.csv", delimiter="\t",
                          index_col=None)
    checkin = pd.read_csv("G:/pyfile/relation_protect/src/data/result_data/comloc/1" + str(k) + "_3_3_comloc.csv",
                          delimiter="\t",
                          names=['uid', 'time', 'latitude', 'longitude', 'locid', 'grid_id', 'clusterid', 'is_core',
                                 'grid_id_before', 'locid_before'], header=None)
    # checkin_obf = pd.read_csv("G:/pyfile/relation_protect/src/data/" + "city_data/1.csv", delimiter="\t",
    #                           index_col=None)
    uids = list(checkin.uid.unique())
    checkin_obf = checkin_obf[checkin_obf.uid.isin(uids)]
    checkin_obf = checkin_obf.values.tolist()
    checkin_obf = grid_divide(checkin_obf, 40, 30, [30.387953, -97.843911, 30.249935, -97.635460]).divide_area_by_NN()

    # checkin_obf = grid_divide(checkin_obf, 40, 40, [40.836357, -74.052914, 40.656702, -73.875168]).divide_area_by_NN()
    # checkin_obf = grid_divide(checkin_obf,  40, 30, [37.809524, -122.520352, 37.708991, -122.358712]).divide_area_by_NN()
    checkin_obf['grid_id'] = checkin_obf['grid_id'].astype(int)  # 将原始数据中的grid_id转换成int类型
    print("原始数据读取成功")
    # commutity_se = commutity_security("comloc", k, "SNAP_NY_1", 0.0044966 * 2, 0.0059352 * 2)
    # commutity_se = commutity_security("comloc", k, "SF", 0.004492 * 2, 0.005681 * 2)
    # commutity_se = commutity_security("comloc", k, "FS_NY_1", 0.0044966 * 2, 0.0059352 * 2)
    # commutity_se = commutity_security("comloc", k, "1", 0.004492 * 2, 0.005202 * 2)
    # commutity_se.set_checkins(checkin, checkin_obf)
    # commutity_se = commutity_security("freqloc", k, "SF", 0.004492 * 2, 0.005681 * 2)
    # commutity_se = commutity_security("freqloc", k, "SNAP_NY_1", 0.0044966 * 2, 0.0059352 * 2)
    # commutity_se = commutity_security("comloc", k, "FS_NY_1", 0.0044966 * 2, 0.0059352 * 2)
    commutity_se = commutity_security("comloc", k, "1", 0.004492 * 2, 0.005202 * 2)
    commutity_se.set_checkins(checkin, checkin_obf)
    commutity_se.cal_security(str(k))
    # commutity_se.cal_utility(0.54, k)
    end = time.time()
    print("总共花的费时间为", str(end-start))