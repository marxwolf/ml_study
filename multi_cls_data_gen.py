#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: Xiang Ma
# @desp: Generate data for multi-classification in machine learning

import csv
import numpy as np

# sample number
N = 100

# x ~ U(0, 10)
x = 10 * np.random.rand(N)
y = np.log2(1 + x)

# make classification of data
z = []
for i in y:
	if i < 1:
		z.append(0.5)
	elif (i > 1 or i == 1) and i < 2:
		z.append(1.5)
	elif (i > 2 or i == 2) and i < 3:
		z.append(2.5)
	else:
		z.append(3.5)

# write to data_smaple file
with open('./cls_data.csv', 'w') as f:
	writer = csv.writer(f, delimiter=',')
	writer.writerows(zip(x, z))
f.close()
