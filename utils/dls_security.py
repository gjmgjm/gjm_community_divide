#!/usr/bin/env python
# encoding: utf-8

"""
@version: v1.0
@author: jaculamata
@license: Apache Licence
@contact: 819436557@qq.com
@site: http://blog.csdn.net/hqzxsc2006
@software: PyCharm
@file: 5_security.py
@time: 2019/1/20 15:44
"""
import pandas as pd
import time
from grid_divide import grid_divide
from sys import argv

class dls_security():

    def __init__(self, city, m, n, range, lats_per_km, lons_per_km):
        self.checkins_obf = pd.read_csv("G:/pyfile/relation_protect/src/data/city_data/"+city+".csv", delimiter="\t", index_col=None)
        # self.m = 40  # 奥斯汀
        # self.n = 30
        # self.range = [30.387953, -97.843911, 30.249935, -97.635460]
        # self.m = 40    # SF
        # self.n = 30
        # self.range = [37.809524, -122.520352, 37.708991, -122.358712]
        # self.m = 40  # NY
        # self.n = 40
        # self.range = [40.836357, -74.052914, 40.656702, -73.875168]
        self.m = m
        self.n = n
        self.range = range
        self.lons_per_km = lons_per_km
        self.lats_per_km = lats_per_km
        self.city = city
        # self.checkins_obf = self.checkins_obf.values.tolist()
        # self.checkins_obf = grid_divide(self.checkins_obf,  30, 40, [30.387953, -97.843911, 30.249935, -97.635460]).divide_area_by_NN()
        # self.checkins_obf['grid_id'] = self.checkins_obf['grid_id'].astype(int)   # 将原始数据中的grid_id转换成int类型
        print("原始数据读取成功")

    def cal_security(self, k, method):
        from security_unility import security_unility
        print(k)
        security = security_unility(self.lons_per_km, self.lats_per_km)
        # checkin = pd.read_csv("G:/pyfile/relation_protect/src/data/result_data/dls/simple" + str(k) + "_ano_HCheckins.csv", delimiter="\t", names=["uid", "locid"], header=None)
        checkin = pd.read_csv("G:/pyfile/relation_protect/src/data/result_data/dls/simple_" + self.city + "_" + str(k) + "_ano_HCheckins2.csv", delimiter="\t", names=["uid", "time", "latitude", "longitude", "locid"], header=None)
        security.set_checkins(checkin, self.checkins_obf, "G:/pyfile/relation_protect/src/data/city_data/", self.city)
        security.set_grid(self.m, self.n, self.range)
        # b = security.b_security(3)
        # c = security.c_security(100)
        # d = security.d_security()
        # e = security.e_security()
        # f = security.f_security1(k)
        # g = security.g_security1(k)
        a = security.unility()
        file = open("G:/pyfile/relation_protect/src/data/result_data/" + method + ".txt", 'a', encoding='UTF-8')
        # file.write('比例参数:' + str(k) + ' ' + 'b:' + str(b) + ' ' + 'c:' + str(c) +
        #            ' ' + 'd:' + str(d) + ' ' + 'e:' + str(e) +
        #            ' ' + 'f:' + str(f) + ' ' + 'g:' + str(g) +
        #            '\n')
        file.write('a:' + str(k) + ' ' + str(a) + '\n')
        file.close()
        # a = self.a_security()
        # self.all_security(a, b, c, d, e, 1/3, 1/3, 1/3)

    def cal_utility(self, k, method):
        from utility import utility
        print(k)
        utility = utility()
        checkin = pd.read_csv("G:/pyfile/relation_protect/src/data/result_data/dls/simple" + str(k) + "_ano_HCheckins2.csv",
                              delimiter="\t", names=["uid", "time", "latitude", "longitude", "locid"], header=None)
        utility.set_checkins(checkin, self.checkins_obf, "G:/pyfile/relation_protect/src/data/city_data/",
                             "G:/pyfile/relation_protect/src/data/result_data/", "1", str(k)+"_ano_", "dls")
        # utility.checkin_sim_list2()
        k_utility = utility.sim_utility2()
        file = open("G:/pyfile/relation_protect/src/data/result_data/" + method + ".txt", 'a', encoding='UTF-8')
        file.write('比例参数:' + str(k) + ' ' + 'utility:' + str(k_utility) + '\n')
        file.close()


if __name__ == "__main__":
    start = time.time()
    k=argv[1]
    # for k in [3, 4, 5, 6, 7, 8, 9, 10]:
    # dls_se = dls_security("SNAP_NY_1", 40, 40, [40.836357, -74.052914, 40.656702, -73.875168], 0.0044966 * 2, 0.0059352 * 2)
    # dls_se = dls_security("SF", 40, 30, [37.809524, -122.520352, 37.708991, -122.358712], 0.004492 * 2, 0.005681 * 2)
    dls_se = dls_security("FS_NY_1", 40, 40, [40.836357, -74.052914, 40.656702, -73.875168], 0.0044966 * 2, 0.0059352 * 2)
    # dls_se = dls_security("1", 40, 30, [30.387953, -97.843911, 30.249935, -97.635460], 0.004492 * 2, 0.005202 * 2)
    dls_se.cal_security(k, "dls_result")
    # dls_se.cal_utility(k, "dls_result")
    end = time.time()
    print("总共花的费时间为", str(end-start))
