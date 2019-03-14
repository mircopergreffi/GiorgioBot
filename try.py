#!/usr/bin/python3

import sys
from classify import classify

if (len(sys.argv) > 1):
	results = classify(sys.argv[1])
	if (len(results)>0):
		print("%s" % results[0][0])
else:
	print("Usage: ./try <phrase-to-say>")