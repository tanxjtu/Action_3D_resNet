'''
    author: Haoliang Tan
    Data: 22/1/2019
'''
import numpy as np


#   We use Charades as our base dataset however we only use some dataset
#   that is to say the dataset is a subset of original Charades

original_class_num = 157
# original = [i for i in range(original_class_num)]

Original_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
                 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
                 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
                 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
                 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145,
                 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156]
# original Charades class num

Part_class_list = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                   26, 27, 28, 29, 30, 31, 32, 46, 47, 48, 49, 50, 51, 52, 59, 60, 106,
                   107, 108, 109, 110, 111, 118, 119, 120, 121, 122, 123, 131, 132, 133,
                   134, 135, 142, 143]

# Sub class that interact with human


def Class_Mapping(fm_label, original_class=Original_list, Part_class_list=Part_class_list):
    return fm_label


def Class_num(Original_list=Original_list, Part_class_list=Part_class_list):
    raw_class_num = len(Original_list)
    part_class_num = len(Part_class_list)
    return raw_class_num, part_class_num, Original_list, Part_class_list