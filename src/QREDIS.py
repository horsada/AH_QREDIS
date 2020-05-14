#QREDIS
# QREDIS startup script
# run this when Python opens to load stuff needed; it should not be imported.
# JAH 20120915 

print("~~~~~~~~~~~~~~~~~~~~~~~~ Welcome to QREDIS 1.0 ~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~ J. Andrew Howe, PhD ~~~~~~~~~~~~~~~~~~~~~~~~")
print("...loading Python modules...")

crit = False 					# critical module not loaded

# base python
try:
	from imp import reload		# get the reload function
except:
	print("\tCan't load function: reload!")
try:
	import sys					# operating system functions
except:
	print("\tCan't load module: sys!")
try:
	import os					# operating system functions
except:
	print("\tCan't load module: os!")
try:
	import string				# string manipulation
except:
	print("\tCan't load module: string!")
	crit=True
try:
	import math					# duhh
except:
	print("\tCan't load module: math!")
	crit=True
try:
	import datetime as dat		# date objects and arithmetic
except:
	print("\tCan't load module: datetime!")
	crit=True
try:
	import time				# time control
except:
	print("\tCan't load module: time!")
	crit=True
try:
	import pdb					# debugging
except:
	print("\tCan't load module: pdb!")
try:
	import inspect as ins		# inspection
except:
	print("\tCan't load module: inspect!")
	crit=True

# extra python packages
try:
	import numpy as np
	import numpy.random as rnd			# random number generators
	from numpy import linalg			# linear algebra functions
	np.set_printoptions(threshold = np.inf)
except:
	print("\tCan't load module: numpy!")
	crit=True
try:
	import scipy
	import scipy.stats as sstat			# statistics functions
except:
	print("\tCan't load module: scipy!")
try:
	import matplotlib.pyplot as plt
	plt.ion()
except:
	print("\tCan't load module: matplotlib!")
	crit = True
	
print("...loading QREDIS modules...")

# change dir,
os.chdir(os.getcwd()+'/QREDIS')

# QREDIS packages
try:
	import QREDIS_Basic as QB
except:
	print("\tCan't load module: QREDIS_Basic!")
	crit = True
try:
	import QREDIS_Data as QD
except:
	print("\tCan't load module: QREDIS_Data!")
	crit = True
try:
	import QREDIS_GA as QG
except:
	print("\tCan't load module: QREDIS_GA!")
try:
	import QREDIS_InfComp as QI
except:
	print("\tCan't load module: QREDIS_InfComp!")
try:
	import QREDIS_Model as QM
except:
	print("\tCan't load module: QREDIS_Model!")
	crit = True
	
# set up some directories JAH 20140118 put in __builtins__ to make it global'ish
# I don't really like this so much, but is seems the only (or best) way to do this
__builtins__['QREDIS_out'] = os.getcwd()+'/out'
QREDIS_out = __builtins__['QREDIS_out']
__builtins__['QREDIS_mod'] = os.getcwd()+'/models'
QREDIS_mod = __builtins__['QREDIS_mod']
__builtins__['QREDIS_dat'] = os.getcwd()+'/Data'
QREDIS_dat = __builtins__['QREDIS_dat']

# talk a little
print('....Current directory set to %s\n....QREDIS_out set to %s\n....QREDIS_mod set to %s\n....QREDIS_dat set to %s'%(os.getcwd(),QREDIS_out,QREDIS_mod,QREDIS_dat))

# finally finally, load base model class 20130408 JAH
exec(open(QREDIS_mod+'/QMod_Template.py').read())

if crit == True:
	print("!!!Some critical modules were not loaded. QREDIS may not work correctly!!!")
print("We can not command discovery, but we can command commitment. - HAPPY MODELING!")
