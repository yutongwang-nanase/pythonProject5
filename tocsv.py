import json
import csv

"""
需求：将json中的数据转换成csv文件
"""
def csv_json():
    # 1.分别 读，创建文件
    json_fp = open("C:/Users/wangyutong/Desktop/jdft2d.json", "r",encoding='utf-8')
    csv_fp = open("C:/Users/wangyutong/Desktop/word.csv", "w",encoding='utf-8',newline='')

    # 2.提出表头和表的内容
    data_list = json.load(json_fp)
#   sheet_title = data_list['index']
    # sheet_title = {"姓名","年龄"}  # 将表头改为中文
    data_energy = []
    sheet_data = []
    i = 0
    for entry in data_list['data']:
        # 创建一个 Structure 对象
        data_energy = [entry[1]]
        cif = i
        sheet_data.append()

        i = i + 1

    for data in data_list:
        sheet_data.append(data.values())

    # 3.csv 写入器
    writer = csv.writer(csv_fp)

    # 4.写入表头
    writer.writerow(sheet_title)

    # 5.写入内容
    writer.writerows(sheet_data)

    # 6.关闭两个文件
    json_fp.close()
    csv_fp.close()


if __name__ == "__main__":
    csv_json()
