#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: Xiang Ma
# @desp: Generate data for regression in machine learning

import csv
import numpy as np

# sample number
N = 100

# x ~ U(0, 100)
x = 100 * np.random.rand(N)
y = np.log(1 + x)

# write to data_smaple file
with open('./reg_data.csv', 'w') as f:
	writer = csv.writer(f, delimiter=',')
	writer.writerows(zip(x, y))
f.close()
