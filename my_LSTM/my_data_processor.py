# coding:utf-8

import numpy as np
import time
from theano import config
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
    for index, line in enumerate(array):
        # 从达到 train_x_length 之后，才开始建立数据
        if index < train_x_length - 1:
            continue
        x_data = array[index-(train_x_length-1): index+1, ]
        # 如果一段训练数据，总时长超过事先的设定值，则丢弃
        if int(x_data[-1, 0]) - int(x_data[0, 0]) > max_time:
            # print(index)
            continue
        # 剩余数组的长度，不够就算目标值了，结束
        if index >= array_length - compute_y_length - 1:
            break
        # 截取用于计算目标值的数据,包含最后一个x值
        y_array = array[index: index+compute_y_length+1, ]
        # 如果一段用于计算目标值的数据，总时长超过事先的设定值，则丢弃
        if int(y_array[-1, 0]) - int(y_array[0, 0]) > max_time:
            continue
        # 截取价格数据
        y_list = np.asarray(y_array[:, 1]).astype(config.floatX)
        # 计算目标值
        y_result = compute_target_from_y_array(y_list)
        y_data = y_result - float(x_data[-1, 1])
        # print(index, y_data, y_list)
        train_x_array.append(x_data)
        train_y_array.append(y_data)
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
