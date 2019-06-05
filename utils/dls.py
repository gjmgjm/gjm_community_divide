# encoding=utf-8
import logging
import math
import random
import sys
import os
from datetime import datetime
from functools import reduce
from itertools import repeat
from operator import add, mul
import pandas as pd
from dls_security import dls_security
from sys import argv


class DLS:
    def __init__(self, k, city, enhanced=True):
        self.d = pd.read_csv("G:/pyfile/relation_protect/src/data/city_data/" + city + ".csv", delimiter="\t", index_col=None)
        self.checkins_tablename = "HoustonCheckins"
        self.ano_checkins_tablename = str(k) + "_ano_HCheckins"
        self.k = k
        self.m = self.k  # M = K
        # self.lons_per_km = 0.005681 * 2  # 旧金山
        # self.lats_per_km = 0.004492 * 2  # delta latitudes per kilo meter
        self.lons_per_km = 0.0059352 * 2  # NY
        self.lats_per_km = 0.0044966 * 2  # delta latitudes per kilo meter
        # self.lons_per_km = 0.005202 * 2  # delta longitudes per kilo meter AS
        # self.lats_per_km = 0.004492 * 2  # delta latitudes per kilo meter
        self.cdt_num = 4 * self.k if enhanced else 2 * self.k  # 候选位置数量
        self.half_cdt_num = 2 * self.k if enhanced else self.k  # 候选数量的一半
        self.ano_locs = []  # distinct anonymous locations匿名位置列表
        self.ano_checkins = []  # anonymous checkins匿名签到列表
        self.ano_checkins1 = []  # 只包含用户id和网格id的匿名签到数据
        self.locations = []  # all locations in the map所有的位置列表
        self.frequencies = []  # frequencies of locations所有位置的频率
        self.grid_ids = []  # grid_ids of locations所有位置对应的网格
        self.ano_checkins2 = []
        self.__init_locs_and_freqs()
        self.__init_logger()
        self.city = city

    # 将真实位置从匿名数据集中识别出来的熵
    @staticmethod
    def entropy(probabilities):
        return -reduce(add, map(lambda x: math.log2(x) * x, probabilities))

    #频率分布转化成概率分布列表
    @staticmethod
    def freqs2probs(freqs):
        freqs_sum = sum(freqs)
        return [freq / freqs_sum for freq in freqs]

    @staticmethod
    def uniformly_random_choice(elements):
        rand = random.uniform(0, sum(elements))  # 产生在0到sum之间的随机数
        first_i_elements_sum = 0
        for i, element in enumerate(elements):
            first_i_elements_sum += element
            if first_i_elements_sum >= rand:
                return i
        return 0

    # 获取地图中按频率升序排序的所有位置
    # 将所有记录的位置处理成一个（纬度，经度）的位置列表，按照（纬度，经度，locid，出现次数）组成频率列表
    def __init_locs_and_freqs(self):
        # results = self.d.groupby(by=['gridlat', 'gridlon', 'grid_id']).size().reset_index(name='freq').sort_values(by=['freq'], ascending=True).reset_index(drop=True)
        results = self.d.groupby(by=['locid', 'latitude', 'longitude']).size().reset_index(name='freq').sort_values(
            by=['freq'], ascending=True).reset_index(drop=True)
        self.locations = [[float(row[2]), float(row[1])] for row in results.itertuples(name=False, index=False)]   #纬度、经度，次数
        assert len(self.locations) > self.cdt_num, '# of locations <= ' + self.cdt_num
        self.grid_ids = [row[0] for row in results.itertuples(name=False, index=False)]
        self.frequencies = [row[3] for row in results.itertuples(name=False, index=False)]

    # 日志初始化
    def __init_logger(self):
        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(level=logging.INFO)
        logfileName = 'DLS_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.log'
        print(os.path.abspath('.'))
        logfilePath = '.\\logs\\' + logfileName
        handler = logging.FileHandler(logfilePath)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.__logger.addHandler(handler)
        self.__logger.info(self.__class__.__name__)
        self.__logger.info(self.checkins_tablename)

    # 欧氏距离计算
    def __euclidean_distance(self, loc1, loc2):
        # ignore the sqrt
        return ((loc1[0] - loc2[0]) / self.lons_per_km) ** 2 + ((loc1[1] - loc2[1]) / self.lats_per_km) ** 2

    # 欧式距离之乘积
    def __distances_product(self, cdt_loc, ano_loc):
        return reduce(mul, map(self.__euclidean_distance, repeat(cdt_loc), ano_loc))

    def __freq_plus_1(self, loc):
        idx = self.locations.index(loc)
        self.frequencies[idx] += 1
        # keep the asending order in frequencies
        # 保持频率的顺序
        for i in range(idx + 1, len(self.locations)):
            if self.frequencies[i] < self.frequencies[i - 1]:
                # If current frequency is less than previous frequence, swap them with cooresponding locations.
                # 若当前频率小于之前的频率，交换两者相互的位置，频率
                self.frequencies[i], self.frequencies[i - 1] = self.frequencies[i - 1], self.frequencies[i]
                self.locations[i], self.locations[i - 1] = self.locations[i - 1], self.locations[i]
                self.grid_ids[i], self.grid_ids[i - 1] = self.grid_ids[i - 1], self.grid_ids[i]
            else:
                break  # Else, current frequency doesn't break the ascending order of self.frequencies, stop.
                # 若不小于之前频率则跳出循环

    # 返回最大熵的位置列表
    def __max_entropy_location_set(self, loc):
        """
        Get candidate locations according to query frequencies from all locations.
        根据来自所有位置的查询频率获取候选位置
        :param loc: 1x2 list, real location.参数位置的横纵坐标，候选数量cdt_num为匿名化k的4倍
        :return maxentropy_locset: #x2 list, note that # of candidate locations <= cdt_num.
        """
        # Find the index of current location 'loc' in self.locations.当前位置的索引
        # Note that current location should NOT be chose as a candidate location.不要选当前位置为候选位置
        crt_loc_index = self.locations.index(loc)
        crt_frq = self.frequencies[crt_loc_index]

        # Generate cdt_num locations as cdt_locs.
        cdt_locs = []
        cdt_frqs = []
        if crt_loc_index < self.half_cdt_num:
            # If # of candidate locations on the left side of current location is less than candidate halfnum,
            # choose the first cdt_num locations in 'locations' as candidate locations.
            #如果当前位置的小于候选数量的一半（2K）即当前位置左侧位置的数量小于，选择除了这个位置的到cdt_num+1满足cdt_num数量的其他位置为候选列表
            cdt_locs.extend(self.locations[:crt_loc_index] + self.locations[crt_loc_index + 1: self.cdt_num + 1])#将除了当前索引的所有位置添加到列表
            cdt_frqs.extend(self.frequencies[:crt_loc_index] + self.frequencies[crt_loc_index + 1: self.cdt_num + 1])#将除了当前索引的所有位置的访问频率添加到列表
        elif len(self.locations) - 1 - crt_loc_index < self.half_cdt_num:
            # If # of candidate locations on the right side of current location is less than candidate halfnum,
            # choose the last cdt_num locations in all_locs(except loc) as candidate locations.
            #当前位置右侧的数量少于候选位置数量一半则选择后一半满足cdt_num的位置列表
            cdt_locs.extend(self.locations[len(self.locations) - self.cdt_num - 1:crt_loc_index] +
                            self.locations[crt_loc_index + 1:])
            cdt_frqs.extend(self.frequencies[len(self.frequencies) - self.cdt_num - 1:crt_loc_index] +
                            self.frequencies[crt_loc_index + 1:])
        else:
            cdt_locs.extend(self.locations[crt_loc_index - self.half_cdt_num:crt_loc_index] +
                            self.locations[crt_loc_index + 1:crt_loc_index + self.cdt_num - self.half_cdt_num + 1])
            cdt_frqs.extend(self.frequencies[crt_loc_index - self.half_cdt_num:crt_loc_index] +
                            self.frequencies[crt_loc_index + 1:crt_loc_index + self.cdt_num - self.half_cdt_num + 1])
        assert len(cdt_locs) == len(cdt_frqs) == self.cdt_num, '# of candidate locations != ' + self.cdt_num

        # Generate M sets containing half_cdt_num locations, i.e., real location and half_cdt_num - 1 dummy locations.
        # 产生m个（即K个）候选数量一半大小的位置集包含一个真实位置剩下的都是仿造位置
        location_sets = []
        probability_sets = []
        entropy_sets = []
        for i in range(0, self.m):
            # Generate M sets, each contains a real location placed to the first place and
            # candidate_halfnum - 1 dummy locations chosen from candidate_locations.
            random_indices = random.sample(range(0, len(cdt_locs)), self.half_cdt_num - 1)  # 获取候选大小一半数量的随机数
            location_set = [loc] + [cdt_locs[j] for j in random_indices]  # 2k大小的集合
            frequency_set = [crt_frq] + [cdt_frqs[j] for j in random_indices]
            location_sets.append(location_set)  # K次的位置集全部添加到一起等最后返回的时候选取最大的根据索引返回
            probability_sets.append(self.freqs2probs(frequency_set))  # 转化频率为概率列表
            entropy_sets.append(self.entropy(probability_sets[-1]))  # 计算各个概率列表的熵
        return location_sets[entropy_sets.index(max(entropy_sets))]  # 返回熵值最大的位置列表，大小为2K

    # 保存匿名签到数据
    def __save_anonymous_checkins(self):
        save_ano_checkins = pd.DataFrame(self.ano_checkins)
        save_ano_checkins.to_csv("G:/pyfile/relation_protect/src/data/result_data/dls/" + self.city + "_" + self.ano_checkins_tablename + ".csv", header=None, index=None, sep='\t')
        # save_simple_ano_checkins = pd.DataFrame(self.ano_checkins1)
        # save_simple_ano_checkins.to_csv("G:/pyfile/relation_protect/src/data/result_data/dls/simple_" + self.city + "_" + self.ano_checkins_tablename + ".csv", header=None, index=None, sep='\t')
        save_simple_ano_checkins2 = pd.DataFrame(self.ano_checkins2)
        save_simple_ano_checkins2.to_csv("G:/pyfile/relation_protect/src/data/result_data/dls/simple_" + self.city + "_" + self.ano_checkins_tablename + "2.csv", header=None, index=None, sep='\t')

    def __get_anonymous_location(self, loc):
        """
        Generate an anonymous location for a given actual location and number of candidate locations.
        根据实际位置和候选位置数量产生匿名位置，返回含K个位置的位置列表
        :param loc: tuple, an actual location to anonymize.
        :return: None, or a list containing K locations.
        """
        anonymous_loc = [loc]  # 要返回匿名位置列表，包含真实位置数量为K或2K(看是否增强)，但是保证起码有K个匿名化的位置可以选
        loc_set = self.__max_entropy_location_set(loc)  # 返回熵值最大的位置列表
        if not loc_set:
            return None
        for i in range(0, self.k - 1):  # 用K次循环选出K个从熵值最大的位置列表选出的匿名位置集合
            # 取熵值最大的位置列表每一个产生的位置元素 去和 要返回的匿名位置列表的数据（一个列表作为参数传入） 计算欧式距离后 累乘 返回结果
            distance_products = list(map(self.__distances_product, loc_set, repeat(anonymous_loc))) # 即每个熵值最大列表的位置与每个返回的匿名位置计算欧式距离后结果累乘，每个累乘结果组成列表一项、共有N(熵值最大的位置列表的位置总数量)项
            index = self.uniformly_random_choice(distance_products)
            dummy_loc = loc_set[index]
            anonymous_loc.append(dummy_loc)
            loc_set.remove(dummy_loc)
        if len([list(t) for t in set(tuple(_) for _ in anonymous_loc)]) < self.k:
            return None
        return anonymous_loc

    def run(self):
        """
        Main algorithm for the location k-anonymization method 'DLS'.
        Refer to:   Niu, B., et al. Achieving k-anonymity in privacy-aware location-based services.
                    in IEEE INFOCOM 2014 - IEEE Conference on Computer Communications. 2014.
        :return: None.
        DLS算法主要是对用户的签到的位置数据进行一个匿名化的选择过程
        """
        # 1. Get all checkins orderred by datetime.按时间顺序获取所有签到
        checkins = self.d

        # 2. Begin to anonymize all checkins.匿名化签到数据
        checkin_num = len(checkins)
        checkin_cnt = 0
        checkin_chunk_size = int(len(checkins) / 10)
        checkinid = 0
        for row in checkins.itertuples(index=False, name=False):#取每条签到记录进行操作
            checkinid += 1
        # for checkinid, userid, locdatetime, lon, lat in checkins:#取每条签到记录进行操作
            # Print the progress when a chunk of checkins are anonymized.打印匿名一块签到的过程
            # userid, locdatetime, lat, lon, gridid = row[0], row[1], row[6], row[7], row[5]
            userid, locdatetime, lat, lon, gridid = row[0], row[1], float(row[2]), float(row[3]), row[4]
            self.__logger.info(''.join([str(checkin_cnt), '/', str(checkin_num), ', checkin', str(checkinid)]))
            if checkin_cnt % checkin_chunk_size == 0:  # finished the anonymization of a chunk of checkins打印一部分匿名化的结果
                print('%-3d%% work complete.' % (int(checkin_cnt / checkin_chunk_size) * 10))
            # loc = (lon, lat)
            loc = [lon, lat]
            cdt_loc_grid_id = []
            ano_loc = self.__get_anonymous_location(loc)#返回的匿名数据集为K个匿名位置
            if ano_loc is None:  # If failed to anonymize, print checkinid
                self.__logger.error('Failure.')
            else:  # If succeeded to anonymize 'location', get 'ano_loc_id' and generate anonymous checkin匿名化成功返回匿名位置id和匿名签到
                self.__logger.info('Success.')
                # ano_checkin = [checkinid, userid, locdatetime, self.__get_ano_loc_id(ano_loc)]#匿名签到添加了匿名位置在匿名列表中的位置索引
                ano_checkin = [userid, locdatetime]
                for cdt_loc in ano_loc:  # K个匿名位置的每一项
                    cdt_loc_grid_id.append(str(self.grid_ids[self.locations.index(cdt_loc)]))
                    # ano_checkin1 = [userid, str(self.grid_ids[self.locations.index(cdt_loc)])]
                    ano_checkin2 = [userid, locdatetime, cdt_loc[1], cdt_loc[0], str(self.grid_ids[self.locations.index(cdt_loc)])]
                    # self.ano_checkins1.append(ano_checkin1)
                    self.ano_checkins2.append(ano_checkin2)
                    # ano_checkin.append(self.grid_ids[self.locations.index(cdt_loc)])  # 将匿名位置所在的网格id添加到匿名签到
                    # for coordinates in cdt_loc:  # 每个匿名位置的坐标添加到匿名签到
                        # ano_checkin += (coordinates,)
                        # ano_checkin.append(coordinates)
                ano_checkin.append(cdt_loc_grid_id)
                ano_checkin.append(ano_loc)
                self.ano_checkins.append(ano_checkin)  # 将匿名签到添加到匿名签到列表
            # Even if an actual location is not anonymized, add 1 to its frequency because it is visited.
            self.__freq_plus_1(loc)
            checkin_cnt += 1

        # 3. Save anonymous checkins to database.保存匿名签到列表到数据库中
        self.__save_anonymous_checkins()


def main(k):
    # In order to run in the cmd, change current working directory to import package LIST
    sys.path.insert(1, sys.path[0] + '\\..\\..\\')

    # DLS(k, "SF", enhanced=True).run()
    # dls_se = dls_security("SF", 40, 30, [37.809524, -122.520352, 37.708991, -122.358712], 0.004492 * 2, 0.005681 * 2)
    # DLS(k, "SNAP_NY_1", enhanced=True).run()
    # dls_se = dls_security("SNAP_NY_1", 40, 40, [40.836357, -74.052914, 40.656702, -73.875168], 0.0044966 * 2, 0.0059352 * 2)
    DLS(k, "FS_NY_1", enhanced=True).run()
    dls_se = dls_security("FS_NY_1", 40, 40, [40.836357, -74.052914, 40.656702, -73.875168], 0.0044966 * 2, 0.0059352 * 2)
    dls_se.cal_security(k, "dls_result")


if __name__ == '__main__':
    k=int(argv[1])
    main(k)
