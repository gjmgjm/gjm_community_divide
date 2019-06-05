import numpy as np
import sqlite3 as db
import time
import math
from sklearn.cluster import k_means  # 该函数在使用时请使用解释器 kmeansChange
import pandas as pd
from DP_security import DP_security
import matplotlib.pyplot as plt
from sys import  argv
'''下载数据集，返回全部数据列表以及坐标列表'''
def loadDataSet(tableName, dbName):
    '''
    :param tableName: 原始数据表
    :param dbName: 数据库名
    :return: 数据集、坐标集、用户列表
    '''
    conn = db.connect(dbName)
    cursor = conn.cursor()
    sql_select = ' '.join(['SELECT * FROM', tableName, ';'])
    lines = cursor.execute(sql_select).fetchall()
    dataMat = []
    data_lines = []
    user_lists = []
    for line in lines:
        dataMat.append([float(line[3]), float(line[4])])
        data_lines.append(list(line))
        if line[0] not in user_lists:
            user_lists.append(line[0])
    return data_lines, dataMat, user_lists


'''得到全局敏感度'''
def HS_select(data_lines,user_lists):
    #挑选层次敏感度，这里用每个用户的最远距离，从这些值间随机挑选出一个最大值
    matrix_line = pd.DataFrame(data_lines, columns=['userid', 'placeid', 'dtime', 'lon', 'lat', 'semantic'])
    user_spot_dict = {}
    for user in user_lists:
        user_spot_lists = np.array(matrix_line.loc[matrix_line['userid'] == user].loc[:, ['lon', 'lat']])
        max_dist_pow = -1
        for i in range(len(user_spot_lists)-1):
            matrix_i_line = user_spot_lists[i]
            matrix_i_end = user_spot_lists[i+1:]
            matrix_result1 = np.tile(matrix_i_line, (len(user_spot_lists)-i-1, 1))-matrix_i_end
            # matrix_result1[:, 0] = matrix_result1[:, 0]/0.01*500
            # matrix_result1[:, 1] = matrix_result1[:, 1]/0.005*500
            matrix_result1[:, 0] = matrix_result1[:, 0] / 0.005202 * 500
            matrix_result1[:, 1] = matrix_result1[:, 1] / 0.0044966 * 500
            matrix_result1 = (matrix_result1**2).sum(axis=1)
            dist_pow = matrix_result1.max()
            if dist_pow > max_dist_pow:
                    max_dist_pow = dist_pow
        user_spot_dict[str(user)] = max_dist_pow
    user_spot_sort = sorted(user_spot_dict.items(), key=lambda item: item[1], reverse=True)
    return np.sqrt(user_spot_sort[0][1])


'''得到两个坐标间的欧式距离'''
def rough_dist2(lon1,lat1,lon2,lat2):   # 0.0044966  0.0059352
    # dis = np.power((lon1-lon2)/0.01*500, 2)+np.power((lat1-lat2)/0.005*500, 2)
    dis = np.power((lon1 - lon2) / 0.005202 * 500, 2) + np.power((lat1 - lat2) / 0.0044966 * 500, 2)
    return np.sqrt(dis)


'''对所有坐标进行聚类'''
def Dp(dbName, data_lines, dataMat, user_lists, k, p, u, HS, tableStr, tableName_obf):
    print(k)
    #数据库，全部数据集，全部坐标集，k个聚类，p迭代次数,u扰动参数，HS敏感度，添加的城市字段以及扰动参数
    dataSet = np.array(dataMat)
    best_centers, best_labels, best_inertia = k_means(dataSet, n_clusters=k, HS=HS, u=u, max_iter=p, n_jobs=8)
    datarecord_lines = list(data_lines)
    clustAssing = best_labels.tolist()  # 每条记录对应的聚类中心以及距离
    #k_number = 0
    for i in range(len(clustAssing)):
        label = clustAssing[i]
        center = best_centers[label]
        spot = dataSet[i]
        dist = rough_dist2(center[0], center[1], spot[0], spot[1])
        datarecord_lines[i].extend([label, dist])
        #k_number += 1
        #if k_number%10000 == 0:
            #print(datarecord_lines[i])
    print("*****kmeans操作执行完毕！成功返回聚类中心点坐标，以及每条记录所属的类和到聚类中心的距离。**********\n")

    lines_lists = datarecord_lines[:]
    result_lists = datarecord_lines[:]
    weight_dict = {}   # 所有用户对应的聚类的签到频率
    user_sort_dist_record = {}  # 所有用户对应聚类中 根据距离远近对记录进行排序
    user_sort_dist_record_result = {}  # 上面的备份
    for user in user_lists:
        user_weight_dict = {}  # 单个用户对应类以及签到频率
        user_sort_record = {}
        user_sort_record_2 = {}
        #print("two step")
        for j in range(k):
            sort_lists = []
            sort_lists_2 = []
            weight_num = 0
            #print("three step")
            for line in lines_lists:
                if line[0] == user and float(line[6]) == float(j):  # 取列表中每条记录对应的聚类中心标号
                    #print("four step")
                    weight_num += 1
                    sort_lists.append(line)
                    sort_lists_2.append(line)
            user_weight_dict[str(j)] = weight_num
            user_sort_record[str(j)] = sorted(sort_lists, key=lambda line: float(line[7]), reverse=True)  #距离降序排列
            user_sort_record_2[str(j)] = sorted(sort_lists_2, key=lambda line: float(line[7]), reverse=True)
        weight_dict[str(user)] = user_weight_dict
        user_sort_dist_record[str(user)] = user_sort_record
        user_sort_dist_record_result[str(user)] = user_sort_record_2
    for user in user_lists:
        #print('对',user,'的用户进行处理**********************************')
        cluster_number_dict = weight_dict[str(user)]
        for key_val, val_val in cluster_number_dict.items():
            val_num = val_val
            #noise = int(np.random.laplace(np.power(4/(u*max(1,val_num)),k)))  #关于添加噪音的参数需要根据每个用户在每个类具体的记录数量来进行调整
            #print('该用户在该聚类中的记录为',str(val_num))
            noise = np.random.laplace(0, np.power(4 / u, k))
            if val_num != 0 and noise < 0:
                del_list = user_sort_dist_record[str(user)][key_val][0]
                result_lists.remove(del_list)
                val_num -= 1
                del user_sort_dist_record_result[str(user)][key_val][0]
            if val_num != 0 and noise > 0:
                result_lists.append(user_sort_dist_record[str(user)][key_val][-1]) # 实际该变量后面没用到
                val_num += 1
                user_sort_dist_record_result_add = user_sort_dist_record_result[str(user)][key_val][-1]
                user_sort_dist_record_result[str(user)][key_val].append(user_sort_dist_record_result_add)
            weight_dict[str(user)][str(key_val)] = val_num
    #return result_lists,weight_dict,user_sort_dist_record_result #增删记录后的全部数据集，用户每个聚类的权重列表，用户在每个聚类中的记录字典
    print("*******调整每个用户聚类中权重的操作完毕！并且获得了增删记录后的全部数据集，用户每个聚类的权重列表，用户在每个聚类中的记录字典。******")

    cluster_all_record = {}  # 存储了每个聚类里包含的所有记录
    for i in range(k):
        i_cluster_records = []
        for key_user in weight_dict.keys():
            key_user_i_cluster_records = user_sort_dist_record_result[key_user][str(i)]
            for record in key_user_i_cluster_records:
                i_cluster_records.append(record)
        cluster_all_record[str(i)] = i_cluster_records
    conn = db.connect(dbName)
    cursor = conn.cursor()
    tablename = 'dp_result_' + str(k) + '_' + str(p) + '_' + tableStr
    sql_create = ' '.join(['DROP TABLE IF EXISTS ', tablename, ';CREATE TABLE ', tablename,
                           '(userid int, locid TEXT, placeid TEXT, dtime TEXT, lon_ori real, lat_ori real,lon_fal real ,lat_fal real,semantic_ori TEXT,semantic_fal TEXT,dist float);']) #userid,locid,placeid,dtime,lon_ori,lat_ori,lon_fal,lat_fal,semantic_ori,semantic_fal
    cursor.executescript(sql_create)
    number_change_spot_1 = 0
    number_change_spot = 0
    for j in range(k):
        #print(j)
        for record in cluster_all_record[str(j)]:
            record_lon = float(record[3])
            record_lat = float(record[4])
            select_all_spots = [[record_select[3], record_select[4]] for record_select in cluster_all_record[str(j)][:]]
            record_ori_spots = np.tile(np.array([record_lon, record_lat]), (len(select_all_spots), 1))
            distLines = record_ori_spots-np.array(select_all_spots)
            # distLines[:, 0] = distLines[:, 0] / 0.01 * 500
            # distLines[:, 1] = distLines[:, 1] / 0.005 * 500
            distLines[:, 0] = distLines[:, 0] / 0.005202 * 500
            distLines[:, 1] = distLines[:, 1] / 0.0044966 * 500
            distLinesPower = distLines ** 2
            distLinesPower = distLinesPower.sum(axis=1) ** 0.5
            Q = np.tile(np.array([HS]), len(distLinesPower)) - distLinesPower
            PR_fenzi_matrix = u*Q/(8*HS)  # 得到的是该记录与该类中所有记录的指数值
            PR_fenzi_ = np.array([math.exp(fenzi) for fenzi in PR_fenzi_matrix])
            PR_fenmu_ = PR_fenzi_.sum()
            PR_fenzi_ = PR_fenzi_ / PR_fenmu_
            distSortIndex = list(PR_fenzi_.argsort())
            record_candidate_record_property = sorted(PR_fenzi_)
            property = np.random.random()
            property_candidate = 0
            i = 0
            if property == 0:
                i = 0
            else:
                while(property_candidate<property):  #轮盘赌过程
                    property_candidate += record_candidate_record_property[i]
                    i += 1
                if property_candidate >= property:
                    i -= 1
            if str(cluster_all_record[str(j)][distSortIndex.index(i)][1]) != str(record[1]):
                number_change_spot_1 += 1
            if str(cluster_all_record[str(j)][distSortIndex.index(i)][3]) != str(record[3]) or str(cluster_all_record[str(j)][distSortIndex.index(i)][4]) != str(record[4]):
                number_change_spot += 1
            #print("record:",record)
            #print("record_candidate:",record_candidate_record_property)
            dist = rough_dist2(float(record[3]), float(record[4]), float(cluster_all_record[str(j)][distSortIndex.index(i)][3]), float(cluster_all_record[str(j)][distSortIndex.index(i)][4]))
            sql_insert = " ".join(["INSERT INTO", tablename, "VALUES ('%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s')" % (record[0], record[1], cluster_all_record[str(j)][distSortIndex.index(i)][1], record[2], record[3],record[4],cluster_all_record[str(j)][distSortIndex.index(i)][3], cluster_all_record[str(j)][distSortIndex.index(i)][4],record[5], cluster_all_record[str(j)][distSortIndex.index(i)][5], dist)])#userid,locid,placeid,dtime,lon_ori,lat_ori,lon_fal,lat_fal,semantic_ori,semantic_fal.
            cursor.execute(sql_insert)
    conn.commit()
    cursor.close()
    conn.close()
    print("所有操作均结束完毕！并且有", number_change_spot_1, "条数据是因为placeid的改变；", number_change_spot, "条数据被替换了坐标。")
    file = open("G:/pyfile/relation_protect/src/data/result_data/DP.txt", 'a', encoding='UTF-8')
    file.write(str(number_change_spot_1) + ' ' + str(number_change_spot) + ' ')
    file.close()
    # 因为数据增删之后重复数据比较多，所以会存在替换了位置的数据反而比替换了地点ID的数据量还要大
    dp_se = DP_security(tablename, tableName_obf)
    dp_se.security_unility()


def DpFile(tableName, dbName, u, p, k, tableStr):
    '''
    :param fileName:数据表名
    :param dbName: 数据库名
    :param u: 差分隐私参数
    :param p:聚类迭代次数
    :param k:聚类个数
    :param tableStr: 给差分隐私扰动后的数据表添加字符标识
    :return: 差分隐私扰动的带替换距离的结果
    '''
    data_lines, dataMat, user_lists = loadDataSet(tableName, dbName)
    HS = HS_select(data_lines, user_lists)  # 求取全局敏感度
    uStr = ''
    for x in str(u).split('.'):
        uStr += x
    tableStr += uStr
    Dp(dbName, data_lines, dataMat, user_lists, k, p, u, HS, tableStr, tableName)
    # 原始数据库，原始数据集列表，原始数据集坐标列表，用户列表，自行指定聚类个数，自行指定迭代次数，自行指定扰动参数，全局敏感度，自行指定添加的说明字符串
    # 差分隐私扰动完后生成的数据表名 'Ori_Dp_Dist_all_data_cluster_weight_select_result_' + str(k) + str(p)+ '_'+tableStr，用于下一个文件背景扩充使用


if __name__ == '__main__':
    # cluster_num = [31, 17, 27, 33]
    # cluster_num = [15]
    # e1 = float(argv[1])
    cluster_num = [31, 17, 33]
    i = 0
    # city = "Gowalla_AS"
    for city in ["Gowalla_AS", "SF", "FS_NY_1"]:
        for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    # for city in ["Gowalla_AS"]:
    #         print(cluster_num[i])
            time_start1 = time.time()  # 差分隐私函数，需要自定义函数内的多个参数，***************************修改解释器**********************
            # dbNameSF = "F://Document//database//gowalla//Gowalla0202_second.db"
            dbNameSF = "G:/pyfile/relation_protect/src/data/city_data/" + city + ".db"
            DpFile(city, dbNameSF, 0.1, 300, cluster_num[i], city+str(i))
            DpFile(city, dbNameSF, 1, 300, cluster_num[i], city+str(i))
            DpFile(city, dbNameSF, 3, 300, cluster_num[i], city+str(i))
            DpFile(city, dbNameSF, 5, 300, cluster_num[i], city+str(i))
            # DpFile(city, dbNameSF, 7, 300, cluster_num[i], city)
            # DpFile(city, dbNameSF, 8, 300, cluster_num[i], city)
            DpFile(city, dbNameSF, 10, 300, cluster_num[i], city+str(i))
            # DpFile(city, dbNameSF, 12, 300, cluster_num[i], city)
            time_end1 = time.time()  # 一个小时
        i += 1
    print("差分隐私扰动花费的时间是：", (time_end1-time_start1)/60, "分钟。")
