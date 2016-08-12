#coding:utf-8

import numpy as np
import time, os

data_length = 1000

def get_data():
    CurrentPath = os.getcwd()
    file_name = CurrentPath + '\\data\\RB01_20150818.csv'
    return data_from_file(file_name)

def data_from_file(filename):
    train_data = []
    target_data = []
    with open(filename, 'r') as file_object:
        line_number = 1
        previous_price = 0.
        for line in file_object:
            # 除去第一行的标题，以及长度比较小的行（最后几行可能为空）
            if  line_number == 1 or len(line) < 10:
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
                data = [line_list[2], line_list[3], line_list[5], line_list[6], line_list[7],
                        line_list[12], line_list[13], b_s]
                # print(data)
                train_data.append(data)
                if line_number >= 3:
                    target = float(line_list[2]) - previous_price
                    target_data.append(target)
                previous_price = float(line_list[2])

            # 调试用
            if line_number > data_length + 1:
                break
            line_number += 1

    del train_data[-1]
    return np.asarray(train_data), np.asarray(target_data)

########################################################################################
if __name__=='__main__':
    CurrentPath = os.getcwd()
    file_name = CurrentPath + '\\data\\RB01_20150818.csv'
    t = time.time()
    x, y = data_from_file(file_name)

    print(x.shape)
    # print(x)
    print(y.shape)
    # print(y)
    # 提取列表中每个元素的第一项
    # print(np.asarray(x)[:,0])
    print('time used:', time.time() - t)