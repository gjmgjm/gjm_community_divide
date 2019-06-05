#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
import xlwt
from openpyxl.workbook import Workbook

class cal_norm():

    def __init__(self):

        pass
        # list = [[1, 2], [3, 4], [1, 3]]
        # list = pd.DataFrame(list, columns=['a', 'b'])
        # writer = pd.ExcelWriter('G:/pyfile/relation_protect/src/data/result_data/result.xlsx', engine='openpyxl')
        # list.to_excel(writer, sheet_name='sf', index=None)
        # writer.save()
        # writer.close()
        # group_records = list.groupby(by=['a'])
        # for group in group_records:
        #     avg_num = group[1].mean().values.transpose()
        #     print(type(avg_num))
        #     print(avg_num)
        # pass

    def cal_avg_norm(self, city, total_checkins_len):
        records = pd.read_csv("G:/pyfile/relation_protect/src/data/result_data/" + city + ".txt", header=None, sep=" ", names=["method", "k", "comloc", "u_frqloc", "globle_freqloc","globle_locdistbt", "u_locdistbt","change_locnum","change_dis", "user_num", "clusternum"])
        records.drop(['method'], axis=1, inplace=True)
        group_records = records.groupby(by=['k'])
        self.writer = pd.ExcelWriter('G:/pyfile/relation_protect/src/data/result_data/result2_before.xlsx', engine='openpyxl')
        for group in group_records:
            # group[1][:, 'dis'] =
            group[1].loc[:, 'change_dis'] = group[1].apply(lambda x: x['change_dis']/x['change_locnum'], axis=1)
            group[1].loc[:, 'change_locnum'] = group[1]['change_locnum'] / total_checkins_len
            # print(group[1])
            avg_num = pd.DataFrame([list(group[1].mean().values.transpose())], columns=["k", "comloc", "u_frqloc", "globle_freqloc","globle_locdistbt", "u_locdistbt","change_locnum","change_dis", "user_num", "clusternum"])
            # print(avg_num)
            data = pd.read_excel('G:/pyfile/relation_protect/src/data/result_data/result2_before.xlsx', header=None, names=["k", "comloc", "u_frqloc", "globle_freqloc","globle_locdistbt", "u_locdistbt","change_locnum","change_dis", "user_num", "clusternum"])
            data = pd.concat([data, avg_num], ignore_index=True).reset_index(drop=True)
            data.to_excel(self.writer, index=False, header=None)
            self.writer.save()

#         pass
#
    def cal_random_disturb(self, city, total_checkins_len):
        records = pd.read_csv("G:/pyfile/relation_protect/src/data/result_data/" + city + ".txt", header=None, sep=" ", names=["ratio", "comloc", "u_frqloc", "globle_freqloc", "globle_locdistbt", "u_locdistbt", "change_locnum", "change_dis"], encoding='')
        records.loc[:, 'change_dis'] = records.apply(lambda x: x['change_dis'] / x['change_locnum'], axis=1)
        # records.loc[:, 'change_locnum'] = records['change_locnum'] / total_checkins_len
        self.writer = pd.ExcelWriter('G:/pyfile/relation_protect/src/data/result_data/rd_result.xlsx', engine='openpyxl')
        records.to_excel(self.writer, index=False, header=None)
        self.writer.save()

if __name__ == "__main__":

    cal_norm = cal_norm()
    # 社区划分结果
    # cal_norm.cal_avg_norm("as_before", 63937)
    # cal_norm.cal_avg_norm("sf_before", 34672)
    # cal_norm.cal_avg_norm("ny_before", 66346)
    cal_norm.cal_random_disturb("random_disturb", 63937)
