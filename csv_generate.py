#!/usr/bin/python
import re
import sys
import csv

shakes = open("test.txt", "r")
title = ['The execution time of each benchmark in Figure 8']
title2 = ['Normalized Speedup = exec. time of implementation/ exec.time of CUDA Baseline']
benchmark = [' ', 'Mandelbrot', 'Convolution', 'DCT', 'FilterBank', 'BeamForming', 'MatrixMul', 'DES', 'MPE']
pagoda = []
fusion = []
baseline = []
pthread = []

pagoda.append("Pagoda")
baseline.append("CUDA Baseline")
fusion.append("CUDA Fusion")
pthread.append("pthread")

line = shakes.readlines()
cat_count = 0
total = 0
count = 0
for i in range(0,len(line)):	
	temp = line[i].split()
	for k in range(0,len(temp)):
		if(temp[k] == "Time:"):
			total = total + float(temp[k + 1])
			count = count + 1
	if(count == 3):
		cat_count = cat_count + 1
		if(cat_count == 1):
			s = '%.4f' % float(total/3)
			pagoda.append(s)
		if(cat_count == 2):
			s = '%.4f' % float(total/3)
			baseline.append(s)
		if(cat_count == 3):
			s = '%.4f' % float(total/3)
			fusion.append(s)
		if(cat_count == 4):
			s = '%.4f' % float(total/3)
			pthread.append(s)
			cat_count = 0
		count = 0
		total = 0
	
resultFile = open("output.csv", "wb")
wr = csv.writer(resultFile, dialect='excel')
wr.writerow(title)
wr.writerow(title2)
wr.writerow(benchmark)
wr.writerow(pagoda)
wr.writerow(fusion)
wr.writerow(baseline)
wr.writerow(pthread)
