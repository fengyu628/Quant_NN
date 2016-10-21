# coding:utf-8

import numpy as np
import time
from theano import config
import copy
# import os

# 一个训练数据的长度
train_x_length = 10
# 用来判断间隔的交易时间
# 一段训练数据，从头到尾，共漏掉的秒数
inter_time = 2
# 一段训练数据，从头到尾，可容忍的最长秒数
max_time = (train_x_length / 2) + inter_time
# 用来计算目标值的数据长度
compute_y_length = 10
# 一个文件跳过开始的几行数据（撮合阶段，某些数值过大）
start_ignore_length = 10


def csv_file_to_train_data(filename):
    return csv_array_to_train_data(csv_file_to_array(filename))


def compute_target_from_y_array(y_list):
    return np.mean(y_list)


def csv_file_to_array(filename):
    csv_array = []
    with open(filename, 'r') as file_object:
        line_number = 1
        for line in file_object:
            # 除去第一行的标题，以及长度比较小的行（最后几行可能为空）
            if line_number == 1 or len(line) < 10:
                pass
            else:
                # 每个100行，打印行号
                # if line_number %100 == 0:
                #     print(line_number)
                # print(line)
                # csv文件用‘,’ 来做分割
                line_list = line.split(',')
                # 把B（主动买）转换为1，把S（主动卖）转换为-1
                b_s = 1 if line_list[18][0] == 'B' else -1
                # print(b)
                # 把时间转换成时间戳
                time_string = line_list[0] + ' ' + line_list[1]
                time_array = time.strptime(time_string, "%Y-%m-%d %H:%M:%S")
                time_stamp = int(time.mktime(time_array))
                '''
                line_list[2]:成交价
                line_list[3]:成交量
                line_list[5]:增减仓
                line_list[6]:B1价
                line_list[7]:B1量
                line_list[12]:S1价
                line_list[13]:S1量
                '''
                data = [time_stamp, float(line_list[2]), float(line_list[3]), float(line_list[5]),
                        float(line_list[6]), float(line_list[7]), float(line_list[12]), float(line_list[13]), b_s]
                # print(data)
                csv_array.append(data)

            # 调试用
            # if line_number > data_length + 1:
            #     break
            line_number += 1

    return np.asarray(csv_array)


def csv_array_to_train_data(array):
    train_x_array = []
    train_y_array = []
    array_length = len(array)
    if array_length < start_ignore_length + train_x_length + compute_y_length:
        print('too small CSV file')
        return
    for index, line in enumerate(array):
        # 在超过了 start_ignore_length 并达到 train_x_length 之后，才开始建立数据
        if index < train_x_length + start_ignore_length - 1:
            continue
        x_array = copy.copy(array[index-(train_x_length-1): index+1, ])

        # 如果一段训练数据，总时长超过事先的设定值，则丢弃
        if int(x_array[-1, 0]) - int(x_array[0, 0]) > max_time:
            # print(index)
            continue

        # 剩余数组的长度，不够就算目标值了，结束
        if index >= array_length - compute_y_length - 1:
            break

        # 截取用于计算目标值的数据,包含最后一个x值
        y_array = copy.copy(array[index: index+compute_y_length+1, ])

        # 如果一段用于计算目标值的数据，总时长超过事先的设定值，则丢弃
        if int(y_array[-1, 0]) - int(y_array[0, 0]) > max_time:
            continue
        # ------------------------ 对训练数据进行预处理 -----------------------------
        # 各种跟价格有关的变量，同时进行归一化，但不做0均值
        '''
        寻找（成交价、B1价、S1价）中的最大值、最小值，做为每一项归一化的最大值和最小值，因为这三项有非常大的相关性。
        '''
        max_price = np.max(x_array[:, 1])
        max_b1_price = np.max(x_array[:, 4])
        max_s1_price = np.max(x_array[:, 6])
        max_price_all = np.max([max_price, max_b1_price, max_s1_price])
        min_price = np.min(x_array[:, 1])
        min_b1_price = np.min(x_array[:, 4])
        min_s1_price = np.min(x_array[:, 6])
        min_price_all = np.min([min_price, min_b1_price, min_s1_price])
        # 最大值映射为1，最小值映射为0（成交价、B1价、S1价）
        x_array[:, 1] = (x_array[:, 1] - min_price_all)/(max_price_all - min_price_all)
        x_array[:, 4] = (x_array[:, 4] - min_price_all)/(max_price_all - min_price_all)
        x_array[:, 6] = (x_array[:, 6] - min_price_all)/(max_price_all - min_price_all)

        # 各种跟量相关的参数，同时进行归一化，但不做0均值
        '''
        寻找（成交量、B1量、S1量）中的最大值，做为每一项归一化的最大值。因为这三项有非常大的相关性（增减仓也是，但增减仓有正负）。
        不做最小值处理，因为量之间的关系函数是过零点的（纯比例关系，不含偏置），直接进行比例缩放即可。
        '''
        max_vol = np.max(x_array[:, 2])
        max_b1_vol = np.max(x_array[:, 5])
        max_s1_vol = np.max(x_array[:, 7])
        max_vol_all = np.max([max_vol, max_b1_vol, max_s1_vol])
        # 最大值映射为1，但不把最小值映射为0（成交量、增减仓、B1量、S1量）
        x_array[:, 2] /= max_vol_all
        x_array[:, 3] /= max_vol_all
        x_array[:, 5] /= max_vol_all
        x_array[:, 7] /= max_vol_all

        # 截取目标价格数据
        y_price_array = np.asarray(y_array[:, 1]).astype(config.floatX)
        # 计算目标值
        y_target = compute_target_from_y_array(y_price_array)
        # 目标值与最后一个成交价的差值
        y_data = (y_target - y_price_array[0])
        train_x_array.append(x_array)
        train_y_array.append(y_data)
        # print(y_data)
    # 在训练数据中去掉时间戳
    x_result = np.asarray(train_x_array)[:, :, 1:9].astype(config.floatX)
    y_result = np.asarray(train_y_array).astype(config.floatX)
    # print(x_result.shape, y_result.shape)
    return x_result, y_result

########################################################################################

if __name__ == '__main__':
    # CurrentPath = os.getcwd()
    # file_name = CurrentPath + '\\data\\RB01_20150818.csv'
    file_name = '..\\training_files\\fu02_20081203.csv'
    t = time.time()
    x1 = csv_file_to_array(file_name)
    print(x1.shape)
    x, y = csv_array_to_train_data(x1)
    print(x.shape, y.shape)
    # print(x)
    # 提取列表中每个元素的第一项
    # print(np.asarray(x)[:,0])
    print('time used:', time.time() - t)
