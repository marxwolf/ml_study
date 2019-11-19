#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: Xiang Ma
# @desp: Generate data for binary-classification in machine learning

import csv
import numpy as np

# sample number
N = 100

# x ~ U(0, 10)
x = 10 * np.random.rand(N)
y = np.log2(1 + x)

# binary split y, actually when x = 5
mid = np.log2(1 + 5)

# make classification of data
z = []
for i in y:
	if i < mid:
		z.append(0)
	else:
		z.append(1)

# write to data_smaple file
with open('./bi_cls_data.csv', 'w') as f:
	writer = csv.writer(f, delimiter=',')
	writer.writerows(zip(x, z))
f.close()
