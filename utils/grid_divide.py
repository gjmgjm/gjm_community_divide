#!/usr/bin/env python  
# encoding: utf-8  

""" 
@version: v1.0 
@author: jaculamata 
@license: Apache Licence  
@contact: 819436557@qq.com 
@site: http://blog.csdn.net/hqzxsc2006 
@software: PyCharm 
@file: 2_grid_divide.py 
@time: 2019/1/20 15:38 
"""
import decimal
import math
import multiprocessing
import pandas as pd
import time


class grid_divide():

    def __init__(self, checkins, N, M, ranges):
        self.checkins = checkins  # 一个城市的数据
        self.ranges = ranges    # 城市经纬度范围
        self.n = N              # 划分网格数
        self.m = M              #
        self.latInterval = decimal.Decimal.from_float(math.fabs(self.ranges[0] - self.ranges[2]) / self.n)
        self.lngIntetval = decimal.Decimal.from_float(math.fabs(self.ranges[3] - self.ranges[1]) / self.m)
        self.maxlat = decimal.Decimal(self.ranges[0])
        self.minlng = decimal.Decimal(self.ranges[1])
        # maxlat, minlng, minlat, maxlng = self.ranges[0], self.ranges[1], self.ranges[2], self.ranges[3]

    def get_latInterval(self):
        return self.latInterval

    def get_lngInterval(self):
        return self.lngIntetval

    def cal_gridid(self, checkin):
        latitude = decimal.Decimal(checkin[2])
        longitude = decimal.Decimal(checkin[3])
        # i = math.ceil((self.maxlat - latitude) / self.latInterval)
        # j = math.ceil((longitude - self.minlng) / self.lngIntetval)
        i = math.floor((self.maxlat - latitude) / self.latInterval)
        j = math.floor((longitude - self.minlng) / self.lngIntetval)
        gridlat = float(self.maxlat - i * self.latInterval - self.latInterval / 2)
        gridlon = float(self.minlng + j * self.lngIntetval + self.lngIntetval / 2)
        checkin.append(int(i * self.m + j))
        checkin.append(gridlat)
        checkin.append(gridlon)
        return checkin

    # 按照N*N网格进行划分
    def divide_area_by_NN(self):
        checkins = self.checkins
        core_num = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(core_num)
        df = pool.map(self.cal_gridid, checkins)
        pool.close()
        pool.join()
        checkins = pd.DataFrame(df, columns=['uid', 'time', 'latitude', 'longitude', 'locid', 'grid_id', 'gridlat', 'gridlon'])
        # print("该城市网格划分完成")
        return checkins

if __name__ == "__main__":
    columns = ['uid', 'time', 'latitude', 'longitude', 'locid']
    start = time.time()
    checkins = pd.read_csv('G:/pyfile/relation_protect/src/data/city_data/1.csv', delimiter="\t")
    print(len(checkins))
    print(len(checkins.uid.unique()))
    checkins = checkins.values.tolist()
    grid_divider = grid_divide(checkins, 20, [29.816691, -95.456244, 29.679229, -95.286390])
    checkins = grid_divider.divide_area_by_NN()
    results = checkins.groupby(by=['gridlat', 'gridlon']).size().reset_index(name='freq').sort_values(by=['freq'], ascending=True).reset_index(drop=True)
    print(results)

    results1 = checkins.groupby(by=['gridlat', 'gridlon','grid_id']).size().reset_index(name='freq').sort_values(by=['freq'],
                                                                                                      ascending=True).reset_index(
        drop=True)
    print(results1)
    # results = results[results.freq > 20]
    # print(results[0:5])
    # locations = [(float(row[0]), float(row[1])) for row in results.itertuples(name=False, index=False)]   #纬度。经度，次数
    # print(locations[0:5])
    # # print(results)
    end = time.time()
    # print(len(checkins))
    # frequencies = [row[2] for row in results.itertuples(name=False, index=False)]
    # print(frequencies[0:5])
    # print(checkins[0:10])
    print("花费时间为：", str(end-start))
    pass  