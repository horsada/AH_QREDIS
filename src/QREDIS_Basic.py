#QREDIS_Basic
"""
QREDIS basic signal and data manipulation functions
----------------------
Attenuate - 20121010 - attenuate a set of signals such that the most recent is scaled 1 and past is gradually scaled down
Check_Holes - 20130321 - identify days for which a ticker is missing prices
Check_Spikes - 20130325 - identify days on which a ticker moves "too much"
DailyValidate - 20130409 - validate (holes/spikes) a ticker for a specified range
DailyValidateList - 20130410 - validate (holes/spikes) a list of tickers for a specified range
Diary - 20121018 - simultaneously print data to both the console and a specified file
Diff - 20120928 - compute an n-day difference for a set of time series
FillHoles - 20121005 - fill missing data by copying the previous day forward
LagLead - 20120914 - generate n-day lags or leads in a set of time series
MissingDays - 20130321 - create a list of unexpected missing days for a specified ticker
MovingAverage - 20121010 - compute n-day arithmetic or weighted moving averages in a set of time series
PlotTickers - 20121015 - create a time-plot of a set of time series, presumably tickers
PlotwDailyRet - 20140201 - plot a single ticker and it's daily changes for a specified date range
PrintTable - 20121030 - create a nice console-printable table from a matrix of data
QuickPlot - 20130501 - make a nice plot of a single ticker
RegressMe - 20121004 - perform either OLS, LSR, or ridge regression and calculate some statistics
RSI - 20121010 - compute n-day Relative Strength technical indicators for a set of time series
SignCumCount - 20121206 - count cumulative days with daily changes in the same direction
TimeStamp - 20121031 - create a nicely-formatted string timestamp
TrendStrength - 20120928 - generate n-day trend strength indicators for a set of time series
VarSubset - 20121018 - generate indices for all possible subsets up to a number p of variables
Whipsaw - 20121010 - create n-day whipsaw indicators for a set of time series
----------------------
JAH 20140117 everything has been tested and seems to be working fine with python 3
"""

import os
import sys
import math
import string
import numpy as np
import datetime as dat
import time
import scipy.stats as sstat
from numpy import linalg
import matplotlib.pyplot as plt
import QREDIS_Data as QD

this_diary = None		# global variable we will always use for the diary function

def LagLead(data, lags, filltype = 1):
	"""
	Introduce specified lag(s) or lead(s) into individual columns in an array.
	---
	Usage: Ldata = LagLead(data, lags, filltype)
	---
	data: (n,p) data array_like of n observations of p variables (or (n,) vector)
	lags: list or integer number(s) <= shape(data)[0] of spaces to lag (if +) or
		lead (if -); if integer, same lag will be used for all columns
	filltype: 1* = fill with nan, 2 = cycle values
	Ldata: array same size as data, with the elements shifted
	---
	ex: lldata = QB.LagLead(transpose(np.array([range(10),[2*i for i in range(10)]])),[3,-4])
	JAH 20120914
	"""

	# data is ducked JAH 20120920	
	data = np.array(data, copy = False)				# first force to an array
	oldshp = data.shape									# now get the old shape
	data = np.array(data, ndmin = 2, copy = False)	# force to 2d
	if data.shape[0] == 1:								# if 0d/1d forced to 2d, will be (1,n), so transpose
		data = data.T									# this is ok to do, since the reshape will force a copy
	(n,p) = data.shape
	
	# replicate lags into list if necessary JAH 20121101
	if (type(lags) is int):
		lags = [lags]*p

	# ensure proper arguments
	if (type(filltype) is not int) or not(filltype in [1,2]):
		raise ValueError("Fill type must be 1 or 2!: %s"%LagLead.__doc__)
	if (p != len(lags)) or (type(lags) is not list):
		raise ValueError("Lags must be interger or list of same length as columns of data: %s"%LagLead.__doc__)
	if max(np.abs(lags)) >= n:
		raise ValueError("Lags must be < n!: %s"%LagLead.__doc__)
	
	# hate looping, but can't think yet how to do this in python without a loop :-( JAH 20120917
	Ldata = np.zeros(data.shape)
	for dim in range(p):
		Ldata[:,dim] = np.roll(data[:,dim],lags[dim])	# first shift the vector
		if (lags[dim] > 0) and (filltype == 1):
			Ldata[:lags[dim],dim] = np.nan				# fill nan in the front
		elif (lags[dim] < 0) and (filltype == 1):
			Ldata[lags[dim]:,dim] = np.nan				# fill nan in the back

	return np.reshape(Ldata,oldshp)						# reshape (if necessary) to match original shape

def Diff(data, numdays):
	"""
	Measure the n-day differences in an array of time series, for a specified number
	of days. The first (number days - 1) elements are set to 0. If number days is 1,
	you can use np.diff, as it should be faster (make sure to use correct axis).
	This function does the same thing after a forced reshape of the data. You can combine
	the Diff and LagLead functions to create a daily returns array:
	(Diff(data,1) / LagLead(data,1))[1:,:]
	---
	Usage: difference = Diff(data, numdays)
	---
	data: (n,p) data array_like of n observations of p variables (or (n,) vector)
	numdays: integer number of days to lookback, must be < n
	difference: array same size as data, holding the daily differences; the first
		(number days - 1) entries are 0
	---
	ex: dat = np.transpose(np.array([range(1,11),[2*i for i in range(3,13)]]))
		print(QB.PrintTable(100*(QB.Diff(dat,1) / QB.LagLead(dat,1))[1:,:],'%0.2f%%',['x_1','x_2']))
	JAH 20120928
	"""
	
	# force data to be a 2d matrix, even if there is only 1 variables
	data = np.array(data, copy = False)				# first force to an array
	oldshp = data.shape									# now get the old shape
	data = np.array(data, ndmin = 2, copy = False)	# force to 2d
	if data.shape[0] == 1:								# if 1d forced to 2d, will be (1,n), so transpose
		data = data.T									# this is ok to do, since the reshape will force a copy
	(n,p) = data.shape

	# ensure proper arguments
	if (type(numdays) is not int) or (numdays >= n):
		raise ValueError("Number days lookback must be integer < n: %s"%Diff.__doc__)
	
	if numdays == 1:
		# prefill the "lag" array
		lag = np.zeros((n,p),dtype=float)
		# get the 1-day difference
		lag[1:,:] = np.diff(data,axis=0)
	else:
		# get the lags first	
		lag = LagLead(data, numdays)
		# fill in the same data in the first (numdays-1) lags, so we will have 0 when we do the subtraction
		lag[0:numdays,:] = data[0:numdays,:]
		lag = data - lag
		
	# subtract, reshape, and return result
	return np.reshape(lag, oldshp)
	
def TrendStrength(data, numdays):
	"""
	Measure the direction and strength of the most recent period of specified length
	for an array of time series.  The trend strength indicator for each day is measured
	by looking back the specified number of days INCLUSIVE and summing the signs of
	the 1-lag differences. These sums are then normalized by the number of days to give
	a relative strength. The first (number days - 1) elements are set to 0.
	---
	Usage: trend strength = TrendStrength(data, numdays)
	---
	data: (n,p) data array_like of n observations of p variables (or (n,) vector)
	numdays: integer number of days to lookback, must be < n
	trend strength: array same size as data, holding the daily trend strengths;
		the first (number days - 1) entries are 0
	---
	ex: ts = QB.TrendStrength(rnd.rand(100,2), 20)
	JAH 20120928
	"""
	
	# force data to be a 2d matrix, even if there is only 1 variables
	data = np.array(data, copy = False)				# first force to an array
	oldshp = data.shape									# now get the old shape
	data = np.array(data, ndmin = 2, copy = False)	# force to 2d
	if data.shape[0] == 1:								# if 1d forced to 2d, will be (1,n), so transpose
		data = data.T									# this is ok to do, since the reshape will force a copy
	(n,p) = data.shape

	# ensure proper arguments
	if (type(numdays) is not int) or (numdays >= n):
		raise ValueError("Number days lookback must be integer < n: %s"%TrendStrength.__doc__)
	
	# get the sign of each daily change JAH 20121010 (previously used LagLead)
	signs = np.sign(Diff(data,1))		
	#lag = LagLead(data,1)
	#lag[0,:] = data[0,:]
	#signs = np.sign(data - lag)	
	
	# prepare
	updown = np.zeros((n,p), dtype=float)	# specify this as float, since we will div. by an int # of days and we want %s
	w = [1.0/numdays]*numdays					# weight vector for the correlation/convolution
	
	# loop through dimensions JAH 20121010
	for dim in range(p):
		# we use correlate with weights of all 1/numdays, which gives us the "average" sign
		updown[(numdays-1):,dim] = np.correlate(signs[:,dim],w,'valid')
	
	# my first attempt - I'd rather loop through p than n
	#for thisday in range(numdays,n+1):
	#	# for each day, look at the last numdays days (inclusive!!!!) and sum the signs
	#	updown[thisday-1,:] = np.sum(signs[(thisday-numdays):thisday,:],axis=0)	
	# finally normalize by # of days, and return reshaped
	# return np.reshape(updown/numdays,oldshp)
	
	return np.reshape(updown,oldshp)
	
def RegressMe(x, y, rtype='O',ridgeparm=None,smallnlargep = 1):
	"""
	Perform multiple (or simple) linear regression between a set of p explanatory
	variables and a single dependent variables.  This function will calculate,
	based on user-specification, regression types of: Ordinary Least Squares,
	Least Squares Ratio (Akbilgic & Deniz Akinci, 2009), or Ridge regression. If
	ridge regression is chosen, the ridge parameter can either be a single k (for k*I)
	or different k's per variable.  This will not regularize the constant term added to x.
	---
	Usage: coefficients, statistics, inverse design matrix = RegressMe(x, y, rtype, ridge parameters,smallnlargep)
	---
	x: (n,p) array_like of n observations of p independent variables; p can be 1 or _;
		do not include a constant term in x, it will be added
	y: array_like of size n of assumed dependent data; if rtype='L', no element of
		y can be 0
	rtype: 'O'* = OLS, 'L' = LSR, 'R' = ridge regression
	ridgeparm: array_like of p ridge parameters; required if type = 'R';
		if p>1 and this is scalar it will be used for all dimensions
	smallnlargep*: how to treat p > n: 1* = warn & exit; 0 = warn only; -1 = do nothing
	coefficients: array of length p+1 holding intercept and regression coefficients
	statistics: array holding [r square, adjusted r square, SST, dof, SSR, dof, SSE, dof, F test stat, F p-value]
	inverse design matrix: (p+1,p+1) inverse design matrix
	---
	ex: betas,stats,invdes = QB.RegressMe(rnd.rand(20,3),rnd.rand(20,1),'L',1)
	JAH 20121004
	"""	
	
	# ensure proper arguments
	try:
		if rtype not in 'OoLlRr':
			raise ValueError("Regression type must be 'O', 'L', or 'R': %s"%RegressMe.__doc__)
	except TypeError:
		raise ValueError("Regression type must be 'O', 'L', or 'R': %s"%RegressMe.__doc__)
	try:
		smallnlargep = int(smallnlargep)
		if abs(smallnlargep) > 1:
			raise ValueError("Small n larege p flag must be int in [1,0,-1]: %s"%RegressMe.__doc__)
	except TypeError:
		raise ValueError("Small n larege p flag must be int in [1,0,-1]: %s"%RegressMe.__doc__)
	
	# first we deal with y, which we force to a column vector
	y = np.array(y,dtype=float,copy=False,ndmin=1)
	# reshape if necessary to column vector; if y somehow came as (n,p>1), this will make ny != nx and we'll raise an error later
	if y.shape != (y.size,1):
		y = y.reshape((y.size,1))
	ny = y.size; mny = np.mean(y)
	
	# now prepare x: if x is 2d with both >1 it is assumed to be (n,p) if not an error will be raised later
	x = np.array(x,ndmin=2,copy=False)				# force to 2d array (column or row vector) at least
	# if x is a vector, force it to a column vector
	if x.shape[0] == 1:
		x = x.T
	# now x is either a column vector or it's an (n,p) matrix, so add on the 1s this should error only if ny<>nx so they don't line up
	try:
		x = np.hstack((np.ones((ny,1)),x))
	except ValueError:
		raise ValueError("Number rows in x must match y: %s"%RegressMe.__doc__)
	(n,p) = x.shape
	
	# by now, y is an (n,1) column vector and x is an (n,p) matrix - either that or we really f'd up	
	if n != ny:
		raise ValueError("Must have number rows in x must match y: %s"%RegressMe.__doc__)
	# JAH 20140126 check for p>n and handle as instructed
	if n < p:
		if smallnlargep == 1:
			# warn & exit
			print('Warning: n = %d < p = %d!'%(n,p))
			return [np.inf]*p, [np.inf]*10, np.inf
		elif smallnlargep == 0:
			# warn & continue
			print('Warning: n = %d < p = %d, results may be useless!'%(n,p))
	
	# procss the ridge parameter if appropriate
	if rtype in 'Rr':
		# if input is scalar, make it a list of length p-1; if p-1=1, this will be simply [ridge]
		if np.isscalar(ridgeparm):
			ridgeparm = [ridgeparm]*(p-1)			
		# check it's not None and it's of length p-1 - this works now for all *correct* cases (input=scalar,list,array)
		if (ridgeparm[0] is None) or (len(ridgeparm) != p-1):
			raise ValueError("Ridge parameter must be scalar or array_like of length p: %s"%RegressMe.__doc__)
		# build the ridge matrix, if we can
		try:
			ridges = np.zeros((p,p))
			ridges[1:,1:] = np.diag(ridgeparm)
		except:		# if there is an exception, it's probably because ridge is a matrix or something weird
			raise ValueError("Ridge parameter must be scalar or array_like of length p: %s"%RegressMe.__doc__)
	
	# compute the regression coefficients for each method - all the stats later should be the same
	if rtype in 'Oo':		# OLS regression
		# catch singlar matrix error and give message and pass out inf JAH 20121108
		invdes = np.dot(x.T,x)
		if linalg.cond(invdes) > 1/sys.float_info.epsilon:
			print("RegressMe: x.T * x condition number too big; not inverting!")
			return [np.inf]*p, [np.inf]*10, np.inf
		# just in case for some reason condition test passes, but invert fails
		try:
			invdes = linalg.inv(invdes)				# design matrix
		except linalg.linalg.LinAlgError as err:
			print("RegressMe: x.T * x can't be inverted: %s, n = %d, p = %d!"%(err,n,p))
			return [np.inf]*p, [np.inf]*10, np.inf
		hat = np.dot(np.dot(x,invdes),x.T)	# hat matrix
		betas = np.dot(np.dot(invdes,x.T),y)
	elif rtype in 'Ll':		# LSR regression
		# beta calculations from A Novel Regression Approach: Least Squares Ratio, Communications in Statistics,
		# 38:9,1539 - 1545, 2009, Akbilgic & Deniz Akinci
		# catch singlar matrix error and give message and pass out inf JAH 20121108
		invdes = np.dot(x.T,x)
		if linalg.cond(invdes) > 1/sys.float_info.epsilon:
			print("RegressMe: x.T * x condition number too big; not inverting!")
			return [np.inf]*p, [np.inf]*10, np.inf
		# just in case for some reason condition test passes, but invert fails
		try:
			invdes = linalg.inv(invdes)				# design matrix
		except linalg.linalg.LinAlgError as err:
			print("RegressMe: x.T * x can't be inverted: %s, n = %d, p = %d!"%(err,n,p))
			return [np.inf]*p, [np.inf]*10, np.inf
		div = x/y									# this (surprisingly) works by broadcasting y against each column of x
		try:
			fst = linalg.inv(np.dot(div.T,div))
		except linalg.linalg.LinAlgError as err:
			print("RegressMe: x.T * x can't be inverted: %s, n = %d, p = %d!"%(err,n,p))
			return [np.inf]*p, [np.inf]*10, np.inf
		sec = (x/(y**2)).T
		betas = np.dot(np.dot(fst,sec),y)		
	elif rtype in 'Rr':		# ridge regression
		# catch singlar matrix error and give message and pass out inf JAH 20121108
		invdes = np.dot(x.T,x) + ridges
		if linalg.cond(invdes) > 1/sys.float_info.epsilon:
			print("RegressMe: x.T * x+ridge condition number too big; not inverting!")
			return [np.inf]*p, [np.inf]*10, np.inf
		# just in case for some reason condition test passes, but invert fails
		try:
			invdes = linalg.inv(invdes)					# design matrix
		except linalg.linalg.LinAlgError as err:
			print("RegressMe: x.T * x can't be inverted: %s, n = %d, p = %d!"%(err,n,p))
			return [np.inf]*p, [np.inf]*10, np.inf
		hat = np.dot(np.dot(x,invdes),x.T)		# hat matrix
		betas = np.dot(np.dot(invdes,x.T),y)
	
	# compute the usual stats (probably not required, but possibly required: force SSs to floats)
	#ons = np.ones((n,n))
	yhat = np.dot(x,betas)								# predicted values	
	SSR = float(np.sum((yhat - mny)**2))					# regression sum of square: dof = p-1
	SSE = float(np.sum((y - yhat) **2))					# error sum of squares: dof = n-p	
	# above are probably faster	
	#SSR = np.dot(np.dot(y.T, (hat-ons/n)),y)			# regression sum of square: dof = p-1
	#SSE = np.dot(np.dot(y.T,(np.eye(n)-hat)),y)	# error sum of squares: dof = n-p
	SST = SSR + SSE											# total sum of squares: dof = n-1
	rsquare = SSR/SST;										# r^2
	rsquareA = 1 - (SSE/SST)*((n-1.0)/(n-p))				# adjusted r^2
	Fteststat = (SSR/(p-1))/(SSE/(n-p))						# F test statistic for the regression relation	
	Fpvalue = 1 - sstat.f.cdf(Fteststat,p-1,n-p)			# p-value of F test statistic

	return betas,[rsquare,rsquareA,SST,n-1,SSR,p-1,SSE,n-p,Fteststat,Fpvalue],invdes
	
def FillHoles(data):
	"""
	This takes an array of daily prices and fills holes by copying the last price forward.
	If there is a missing price on the first day it will be left.
	---
	Usage: data, filled = FillHoles(data)
	---
	data: input (n,p) data array_like of n observations of p variables (or (n,) vector)
	data: output (n,p) data array same size as input, but with 0s filled
	filled: number holes filled
	---
	ex: x = rnd.random_integers(0,5,(10,4)); xf, filled = QB.FillHoles(x)
	JAH 20121005
	"""	
	
	# duck type to array
	data = np.array(data, copy = False, ndmin=1)
	datshape = data.shape
	
	# check for holes, and if none just pass data back out; this doesn't bother to exclude the first day
	filled = np.sum(data == 0)
	if filled == 0:
		return data, 0
		
	# get the indices of which possible 0s we need to skip
	if (data.ndim < 2) or (1 in datshape):
		excludes = [0]
	else:		
		excludes = range(0, datshape[0]*datshape[1], datshape[0])	# this gives the indices, after flattening, of the first day
		
	# flatten to a vector, which will make it much easier to scan & replace missing 0s, 'F' tells it to go column-by-column
	# if data is already a vector, the kind of flattening (F or C) has no effect
	dataf = data.flatten('F')
		
	# find where we have 0s
	found = np.nonzero(dataf == 0)[0]
	# loop through the missing prices, but skipping those from the very first day
	# would rather just exclude the elements in excludes from found, but don't know how, efficiently	
	for z in found:
		if z not in excludes:
			dataf[z] = dataf[z-1]
	
	return dataf.reshape(datshape,order='F'), (filled - np.sum(dataf == 0))

def Attenuate(data):
	"""
	Attenuate a time series of any data (mostly returns or signals) so that recent data
	is weighted more strongly than past data.  This is done by multiplying each data point
	by it's 1-based index then dividing by the number observations.  Hence, the oldest
	observation is attenuated by 1/n and the most recent by 1 (no attenuation).
	---
	Usage: attenuated data = Attenuate(data)
	---
	data: (n,p) data array_like of n observations of p variables (or (n,) vector)	
	attenuated data: array same size as data, holding the attenuated data
	---
	ex: attdata = QB.Attenuate(0.5-rnd.rand(100,2))
	JAH 20121010
	"""
	
	# force data to be a 2d matrix, even if there is only 1 variables
	data = np.array(data, copy = False)				# first force to an array
	oldshp = data.shape									# now get the old shape
	data = np.array(data, ndmin = 2, copy = False)	# force to 2d
	if data.shape[0] == 1:								# if 1d forced to 2d, will be (1,n), so transpose
		data = data.T									# this is ok to do, since the reshape will force a copy
	(n,p) = data.shape
	
	# create the attenuation vector - essentially a weight vector from 1/n up to 1
	attsignal = np.cumsum(np.ones((n,1),dtype=float),axis=0)/n
	
	# since we know the data is (n,p) and the attenuation vector is (n,1) we can broadcast them together - sweet!
	attdata = data*attsignal
	
	return np.reshape(attdata,oldshp)

def MovingAverage(data, numdays, weighted=False):
	"""
	Measure the direction and strength of the most recent period of specified length
	Compute the moving average in each column of data using a specified number of days.
	If the optional parameter is passed as True, this will compute a weighted moving
	average where each observation is multiplied by it's 1-based index in the moving days
	slice, then the sum of all these is normalized by the sum of the weights:
	(1*x1+2*x2+...+n*xn)/(1+2+...+n).
	---
	Usage: averaged data = MovingAverage(data, numdays, weighted)
	---
	data: (n,p) data array_like of n observations of p variables (or (n,) vector)
	numdays: integer number of days to lookback, must be < n
	weighted: flag indicating True= weighted moving average, False* = regular moving average
	averaged data: array same size as data, holding the averaged data;
		the first (number days - 1) entries are 0
	---
	ex: madata = QB.MovingAverage(rnd.rand(100,2),20,True)
	JAH 20121010
	"""
	
	# force data to be a 2d matrix, even if there is only 1 variables
	data = np.array(data, copy = False)				# first force to an array
	oldshp = data.shape									# now get the old shape
	data = np.array(data, ndmin = 2, copy = False)	# force to 2d
	if data.shape[0] == 1:								# if 1d forced to 2d, will be (1,n), so transpose
		data = data.T									# this is ok to do, since the reshape will force a copy
	(n,p) = data.shape

	# ensure proper arguments
	if (type(numdays) is not int) or (numdays >= n):
		raise ValueError("Number days lookback must be integer < n: %s"%MovingAverage.__doc__)
		
	# build the weight vector for correlation/convolution
	if weighted:
		# WMA, so attenuate
		w = np.cumsum(np.ones(numdays,dtype=float),axis=0)
		w = w/np.sum(w)
	else:
		# regular MA, so everything is weighted evenly - by ducktyping, a list is ok
		w = [1.0/numdays]*numdays
		
	# build the output matrix
	madata = np.zeros(data.shape)
	# correlate won't work column-wise, so we must loop :-(
	for dim in range(p):
		madata[(numdays-1):,dim] = np.correlate(data[:,dim],w,'valid')
		# correlate(data,w) is the same as convolve(data,w[::-1])
		
	return np.reshape(madata,oldshp)

def RSI(data, numdays):
	"""	
	Compute the well-known relative strength indicator for technical analysis, computed
	over a specified moving number of days. Here it ranges from 0 to 1 instead of the
	usual 0 to 100. If the number days used is too small, it is possible there will be
	no movement in either the up or down direction in a period.  If it happens that the
	total moved down in a period is 0, the RSI is force to 1. The RSI is computed as
	1 - 1/(1 + sum(up day returns)/sum(down day returns)).
	---
	Usage: rsi = RSI(data, numdays)
	---
	data: (n,p) data array_like of n observations of p variables (or (n,) vector)
	numdays: integer number of days to lookback, must be < n
	rsi: array same size as data, holding the daily relative strength indicator values;
		the first (number days - 1) entries are 0
	---
	ex: rsidiff = QB.RSI(rnd.rand(100),10) - QB.RSI(rnd.rand(100),20)
	JAH 20121010
	"""
	
	# force data to be a 2d matrix, even if there is only 1 variables
	data = np.array(data, copy = False)				# first force to an array
	oldshp = data.shape									# now get the old shape
	data = np.array(data, ndmin = 2, copy = False)	# force to 2d
	if data.shape[0] == 1:								# if 1d forced to 2d, will be (1,n), so transpose
		data = data.T									# this is ok to do, since the reshape will force a copy
	(n,p) = data.shape

	# ensure proper arguments
	if (type(numdays) is not int) or (numdays >= n):
		raise ValueError("Number days lookback must be integer < n: %s"%RSI.__doc__)
	
	# get the daily changes
	returns = Diff(data,1)
	# split the up days from the down days and we'll use correlate
	updys = np.abs(returns)*(returns>0)		# abs is because this can have -0.'s
	dndys = np.abs(returns)*(returns<0)
	
	# prepare
	rsi = np.zeros((n,p))
	w = [1.0]*numdays						# weight vector for the correlation/convolution
	
	# loop through dimensions
	for dim in range(p):
		# we use correlate with weights of all 1 the sum of up days and sum of down days
		# this is usually shown as RS = MA(up,n)/MA(dn,n), but this is the same as sum(UP)/sum(DN) * n/n so why divide?
		ups = np.correlate(updys[:,dim],w,'valid')
		dns = np.correlate(dndys[:,dim],w,'valid')
		# if dns is 0 for some reason, this will be a ZeroDivide exception, so store inf ...
		dns[dns == 0] = np.inf
		# compute the relative strength indicator
		rsi[(numdays-1):,dim] = 1 - 1.0/(1 + ups / dns)
		# ... if any infs in dns, set the rsi to 1
		rsi[dns == np.inf] = 1
	
	return np.reshape(rsi,oldshp)

def Whipsaw(data, numdays):
	"""	
	Compute a whipsaw indicator that counts how often in each numdays period an up day
	is followed by a down day, and vice versa. The indicator is the count of the
	reversals divided by the number of days. It oscillates in the range [0,1], with 0
	indicating perfect persistance (data is monotonic up or down) and 1 indicating
	perfect whipsaw (direction changes everyday). 1 - whipsaw can be used as a
	persistance indicator
	---
	Usage: wsaw = Whipsaw(data, numdays)
	---
	data: (n,p) data array_like of n observations of p variables (or (n,) vector)
	numdays: integer number of days to lookback, must be < n
	wsaw: array same size as data, holding the daily whipsaw indicator values;
		the first (number days - 1) entries are 0
	---
	ex: wsaws = QB.Whipsaw(rnd.rand(100,2),20)
	JAH 20121010
	"""
	
	# force data to be a 2d matrix, even if there is only 1 variables
	data = np.array(data, copy = False)				# first force to an array
	oldshp = data.shape									# now get the old shape
	data = np.array(data, ndmin = 2, copy = False)	# force to 2d
	if data.shape[0] == 1:								# if 1d forced to 2d, will be (1,n), so transpose
		data = data.T									# this is ok to do, since the reshape will force a copy
	(n,p) = data.shape

	# ensure proper arguments
	if (type(numdays) is not int) or (numdays >= n):
		raise ValueError("Number days lookback must be integer < n: %s"%Whipsaw.__doc__)

	# get the daily directional changes and roll them
	signs = np.sign(Diff(data,1))
	prevday = np.roll(signs,1,axis=0)
	# figure out where the signal reversed - where it reverses, a signal will be 1/-1
	# but the rolled forward signal will be -1/1 so they will sum to 0
	revs = signs + prevday
	# but maybe it was flat 2 days in a row (not likely) so take those out
	revs = (revs == 0)*(prevday != 0)

	# now we get to do the moving count
	wsaws = np.zeros((n,p))
	w = [1.0/numdays]*numdays						# weight vector for the correlation/convolution

	# loop through dimensions :-(
	for dim in range(p):
		wsaws[(numdays-1):,dim] = np.correlate(revs[:,dim],w,'valid')
	
	return np.reshape(wsaws,oldshp)

def PlotTickers(title, names, prices, dates, holidays = None, fill=True, ax=None):
	"""
	Plot up to 5 ticker prices together for the same calendar.  This could be separate
	tickers or Open High Low Close (or whatever) for the same ticker.  It could also
	include anything else reasonable you want, such as transformations, returns, signals,
	etc. If the ax parameter is the string "M", each series will be plotted on a separate
	subplot. Otherwise, they will all be plotted together. Note, though, that all time
	series will be plotted on the same scale.
	---
	Usage: fig = PlotTickers(title, names, prices, dates, holidays, fill, ax)
	---
	title: string title for plot; the date range will be added on the same title line
	names: p-length list of names for the tickers to be printed on the legend and/or title
	prices: (n,p) array_like of n days of prices for p indices
	dates: n-length dat.date array_like of price dates 
	holidays*: optional bool array_like same shape as dates, indicating which are holidays 
	fill*: True* = fill missing/0 days using FillHoles after holidays removed
		False = don't; filled holes will be marked on the curves with a star "*"
	ax*: if this is a string "M", each column of prices will get it's own subplot; if
		if is a axes type, the plot will be created here; axes should be from axes = plt.subplot();
		if None, just creates a new figure and puts them all together
	fig: figure handle
	---
	ex: da,ha = QD.GetTickerCalendar('XU100_US',trim=True)
	d,h,p = QD.GetTickersPrices_L(['XU100_US','SPX','NDX','RUT'],da,'Close')
	QB.PlotTickers('IMKB 100 + US Indices', ['XU100_US','SPX','NDX','RUT'],p,da,h[:,0],True,"M")
	JAH 20121015
	"""
	
	# threshhold for day-mon vs mon-year printing; up to 2 trading months will show daily
	nthresh = 40
	
	# duck the array_likes
	dates = np.array(dates, copy=False)
	prices = np.array(prices, ndmin = 2, copy=False)
	# and force prices to be (n,p)
	if prices.shape[0] == 1:							# if 1d forced to 2d, will be (1,n), so transpose
		prices = prices.T
	(n,p) = prices.shape
	
	# exit immediately if more than 5 tickers
	if p > 5:
		raise ValueError("Number tickers must not exceed 5: %s"%PlotTickers.__doc__)
		
	# check ax = "M" if subplots is true, the multiple colors/markers will not be used,
	# the legend will not be used, and the titles will be different
	subplots = (ax == "M")
	
	# handle holidays being None
	if holidays is None:
		holidays = np.zeros(dates.shape,dtype = bool)
	
	# force names to list
	if type(names) is not list:
		names = [names]
	
	# get the lengths
	try:
		lls = [len(dates), n, len(holidays)]
	except TypeError:
		# if error here that means something wrong with most important arrays
		raise TypeError("Something wrong with dates, prices, or holidays array(s): %s"%PlotTickers.__doc__)
	
	# check all same length
	if not(all([x == lls[0] for x in lls])):
		raise ValueError("Arrays should all be same length: %s"%PlotTickers.__doc__)
	if p != len(names):
		raise ValueError("Number of ticker names does not match number columns of tickers: %s"%PlotTickers.__doc__)	

	# assuming all is ok, get rid of the holidays
	dates = dates[~holidays]
	prices = prices[~holidays,:]
	
	# fill in holes if appropriate
	if fill:
		# first we remember where the holes were so we can mark them on the plot
		holes = prices == 0
		prices = FillHoles(prices)[0]
	
	# get the x values - need n again since we took out holidays potentially
	(n,p) = prices.shape
	x = np.arange(n)	
	
	# prepare the dates - store a list of string representations AND arrays of month/year numbers
	# JAH 20121106 added years array
	dates_string = np.array(['           ']*n,dtype=str)
	if n > nthresh:
		dates_mos = np.zeros(n,dtype=int); dates_yrs = dates_mos.copy()
		for dat in range(n):
			dates_string[dat] = dates[dat].strftime('%d-%b-%Y')
			dates_mos[dat] = dates[dat].month
			dates_yrs[dat] = dates[dat].year
	else:
		for dat in range(n):
			dates_string[dat] = dates[dat].strftime('%d-%m-%y')
				
	# build the color-marker strings - I could setup something to do all combos
	# of these markers and colors, but a plot like that would be unreadable, so
	# only allow 5 tickers on a plot (see above)
	mrks = ['-','--',':','^','o'] # not used, because this is generally only needed for printed plots
	cols = ['k','b','r','g','m']
	
	# build the plot
	if ax is None:
		# JAH 20121119 added optional parameter ax
		fh = plt.figure
		ax = plt.subplot(1,1,1)
	elif subplots:
		fh = plt.figure
		#ax = plt.subplot(p,1,1)
	else:
		fh = plt.gcf();
	
	# prepare the x-axis labels
	if n <= nthresh:
		labs = [dat[:5] for dat in dates_string]
	else:			# otherwise just show months
		# first we find where the months change and get the indices
		newmon = x[dates_mos != np.roll(dates_mos,1,axis=0)]
		newyer = x[dates_yrs != np.roll(dates_yrs,1,axis=0)]
		# now set the ticks and labels to just these indices and just the mm-yy part of the date strings
		# only print the yyyy at the start of each year JAH 20121106
		labs =  np.array([(dates_string[cnt][3]*(cnt in newmon))+\
			('\n'+dates_string[cnt][7:]*(cnt in newyer)) for cnt in range(n)])
		labs = labs[newmon]

	# JAH 20121126 how this is done changes if subplots are used or not
	if subplots:
		for dim in range(p):
			ax = plt.subplot(p,1,dim+1)
			plt.plot(x,prices[:,dim])
			# have to loop through holes to annotate them :-(
			if fill:
				for h in np.where(holes[:,dim])[0]:
					plt.annotate('*',(x[h],prices[h,dim]),color='r')
			plt.xlim([0,x[-1]])	
			# now we want to show the calendar x-axis labels
			if n <= nthresh:
				# now set the ticks and tick labels; the labels are just the dd-mm part of the date strings
				ax.set_xticks(x)
				ax.set_xticklabels(labs,rotation=-45)
			else:			# otherwise just show months
				# now set the ticks and labels to just these indices and just the mm-yy part of the date strings
				# only print the yyyy at the start of each year JAH 20121106
				ax.set_xticks(newmon)
				ax.set_xticklabels(labs)
			if dim == 0:
				plt.title('%s for %s - %s'%(title,dates[0],dates[-1]))
			plt.ylabel('%s'%(names[dim]))
	else:	
		plt.hold(True)
		for dim in range(p):
			plt.plot(x,prices[:,dim],cols[dim])
			# have to loop through holes to annotate them :-(
			if fill:
				for h in np.where(holes[:,dim])[0]:
					plt.annotate('*',(x[h],prices[h,dim]),color='r')
		plt.hold(False)
		plt.xlim([0,x[-1]])
		# now we want to show the calendar x-axis labels
		if n <= nthresh:
			# now set the ticks and tick labels; the labels are just the dd-mm part of the date strings
			ax.set_xticks(x)
			ax.set_xticklabels(labs,rotation=-45)
		else:			# otherwise just show months
			# first we find where the months change and get the indices
			newmon = x[dates_mos != np.roll(dates_mos,1,axis=0)]
			newyer = x[dates_yrs != np.roll(dates_yrs,1,axis=0)]
			# now set the ticks and labels to just these indices and just the mm-yy part of the date strings
			# only print the yyyy at the start of each year JAH 20121106
			ax.set_xticks(newmon)
			ax.set_xticklabels(labs)
		# finish up JAH 20130325 added title input to plt.title
		plt.title('%s for %s - %s'%(title,dates[0],dates[-1]))
		if p > 1:
			plt.legend(names)
	
	plt.show()
	return fh

def VarSubset(p):
	"""
	Generate an array of binary indices that can be used for all-subset combinatorial analysis
	of a dataset with p variables.
	---
	Usage: subset_binaries, subset_sizes = VarSubset(p)
	---
	p: integer indicating number of variables to subset
	subset_binaries: (2^p, p) array of all subsets binary indices that can be used to subset
		into the presumed original data matrix
	subset_sizes: 2^p array indicating number of variables in each subset
	---
	ex: p = 4; cols = np.arange(p); bins,sizs = QB.VarSubset(p); print(cols[bins[8,:]])
	JAH 20121018
	"""
	
	# check that p is int; could just duck-type it, but if user passes something else, something is screwed up
	if type(p) is not int:
		raise ValueError("The number variables must be integer: %s"%VarSubset.__doc__)
	
	# prepare the output array; we want bool, but have to start with int, so the assignment below works correctly
	subbins = np.zeros((2**p,p),dtype=int)
	
	# loop through all subsets :-( getting the binary representations
	for cnt in range(1,2**p):
		# get binary representation into a list, then put it in the array
		tmp = bin(cnt)[2:]
		subbins[cnt,(-len(tmp)):] = list(tmp)
	
	# fill in the variable counts
	subsize = np.sum(subbins,axis=1)
	
	# finally sort by variable counts
	tmp = np.argsort(subsize)
	
	return subbins[tmp,:]==1, subsize[tmp]

class diaryout(object):
	"""
	JAH 20121018: this is a diary class that I wrote looking at input here:
	http://stackoverflow.com/questions/616645/how-do-i-duplicate-sys-stdout-to-a-log-file-in-python
	http://stackoverflow.com/questions/8777152/unable-to-restore-stdout-to-original-only-to-terminal
	it's purpose is to help emulate the Matlab diary function of simultaneously printing to a text
	file and the console screen.
	!!!DO NOT USE BY YOURSELF! ONLY USE THE ASSOCIATED Diary FUNCTION!!!
	"""
	def __init__(self):
		self.terminal = sys.stdout
		self.save = None
		
	def __del__(self):
		try:
			self.save.flush()		
			self.save.close()
		except Exception:
			# do nothing, just catch the error; maybe it was instantiated, but never opened
			# this has been removed in Python 3 JAH 20140103
			#sys.exc_clear()
			1/1
		self.save = None
	
	def dclose(self):
		self.__del__()
		
	def __getattr__(self, name):
		if name != 'write':
			return self.terminal.__getattribute__(name)		
	
	def write(self, message):
		self.terminal.write(message)
		self.save.write(message)
	
	def dopen(self,outfile):
		self.outfile = outfile
		try:
			self.save = open(self.outfile, "a")
		except Exception as e:
			# just pass out the error here so the Diary function can handle it
			raise e

def Diary(outfile = None):
	"""
	Simultaneously print output (using python print command) to both the usual screen
	output AND a specified file.  This emulates the diary() function in Matlab; including
	that if the file exists, new output is appended.
	---
	Usage: Diary(outfile)
	---
	outfile: if None* turns off diary output, if '?', this returns the 
	current diary file, otherwise, this is the file it gets saved to
	---
	ex: QB.Diary('/home/ahowe42/out%s.QRE'%string.replace(str(dat.datetime.now())[:10],'-',''))
	JAH 20121018
	"""
	global this_diary
	
	if outfile is None:
		# None passed, so close the diary file if one is open
		if isinstance(this_diary, diaryout):
			sys.stdout = this_diary.terminal	# set the stdout back to stdout
			this_diary.dclose()					# flush and close the file
			this_diary = None					# "delete" it
	elif outfile == '?':
		# just tell the current outfile
		if isinstance(this_diary,diaryout):
			return this_diary.outfile
		else:
			return None
	else:
		# file passed, so let's open it and set it for the output			
		this_diary = diaryout()					# instantiate
		try:
			this_diary.dopen(outfile)			# open & test that it opened
		except IOError:
			raise IOError("Can't open %s for append!"%outfile)
			this_dairy=none						# must uninstantiate it, since already did that
		except TypeError:
			raise TypeError("Invalid input detected - must be string filename or None: %s"%Diary.__doc__)	
			this_dairy=none						# must uninstantiate it, since already did that
		sys.stdout = this_diary					# set stdout to it

def PrintTable(data,formats,colheads,rowheads = None):
	"""
	This function takes in a table of numbers, and returns a nicely formatted character
	string table that can be printed to the console. To directly manipulate the table
	rows, use tablist = string.split(table,'\n').
	---
	Usage: table = PrintTable(data,format,colheads,rowheads)
	---	
	data: (nxp) array of data
	formats: (px1) array_like of sprintf style column formats; if single string is input, every
		column is formatted identically
	colheads: (px1) array_like of column headings
	rowheads*: (nx1) optional array_like of row headings
	table: string of formatted table
	---
	ex: print(QB.PrintTable(rnd.rand(5,3),'%0.2f',['1','ii','three'],['one','two','three','four','five']))
	JAH 20121030
	"""	

	# unlike most of QREDIS_Basic, it doesnt make sense to allow input of not (n,p) array (even if n or p is 1)
	try:
		(datr,datc) = data.shape
	except Exception as e:
		raise e("Variable data must be a 2-d array: %s"%PrintTable.__doc__)
	
	# replicate formats if needed
	if type(formats) is str:
		formats = [formats]*datc
	# and make blank rowheads if needed: do this because it makes the code simpler
	if rowheads is None:
		rowheads = ['']*datr
	
	# check inputs
	try:
		# check column headers - can be either array or list, should make no difference
		if len(colheads) != datc:
			raise ValueError("The number of column headers must match number columns of data: %s"%PrintTable.__doc__)
		# check format strings - can be either array or list, should make no difference
		if len(formats) != datc:
			raise ValueError("The format strings must be 1 or match number columns of data: %s"%PrintTable.__doc__)
		# check row headers - can be either array or list, should make no difference
		if len(rowheads) != datr:
			raise ValueError("The number of row headers must match number rows of data: %s"%PrintTable.__doc__)	
	except TypeError:
		# if it comes here, it's probably because an input is not array_like i.e. len() didn't work; this will cause a typeerror
		raise TypeError("Something wrong with one of the inputs: %s"%PrintTable.__doc__)	
	
	# get lengths
	lens = np.zeros((datr+1,datc),dtype=int)
	# get lengths of column headings
	lens[0,:] = [len(ch) for ch in colheads]
	# now get the lengths of all the data
	for rcnt in range(datr):
		lens[rcnt+1,:] = [len(("%s"%formats[ccnt])%data[rcnt,ccnt]) for ccnt in range(datc)]
	# get the max lengths for each column + 1 extra column for each variable except last
	maxes = np.max(lens,axis=0)+1; maxes[-1] = maxes[-1] - 1
	extras = maxes - lens
	
	# prepare for row headings, if appropriate
	rhbars = ''
	rhclhd = ''
	if rowheads is not None:
		# get the lens and max
		rhlens = [len(rh) for rh in rowheads]
		rhmax = np.max(rhlens)+1
		# build the bar & column header space
		rhbars = '-'*rhmax
		rhclhd = ' '*rhmax
		# left-justify the row headers
		rowheads = [rowheads[rcnt]+' '*(rhmax-rhlens[rcnt]) for rcnt in range(datr)]
	
	# now build up the table string
	bars = '-'*np.sum(maxes) + rhbars
	tablestr = bars
	# first add the column headers
	thisrow = rhclhd
	for ccnt in range(datc):
		# pad is created by div'ing the extra space by 2 (if odd, rounds down) and adding one ...
		# JAH 20121221, in preparation for future use of 3.x, force integer division with //, since floor
		# division will no longer be the default		
		pad = ' '*(extras[0,ccnt]//2+1)
		# ... then taking [1:maxes+1] of the result we should get the exact length column with
		# even pads or 1 extra on the right if required DAMN I'm gooooood JAH
		thiscol = ('%s%s%s'%(pad,colheads[ccnt],pad))[1:(maxes[ccnt]+1)]
		# append the column to the row
		thisrow = thisrow + thiscol
	# append the row to the table
	tablestr +='\n'+thisrow+'\n'+bars
	
	# now do the exact same thing, but for the columns of the data
	for rcnt in range(datr):
		thisrow = rowheads[rcnt]
		for ccnt in range(datc):
			pad = ' '*(extras[rcnt+1,ccnt]//2+1)
			thiscol = ('%s%s%s'%(pad,("%s"%formats[ccnt])%data[rcnt,ccnt],pad))[1:(maxes[ccnt]+1)]
			# append the column to the row
			thisrow = thisrow + thiscol
		# append the row to the table
		tablestr +='\n'+thisrow
	tablestr+='\n'+bars
	
	return tablestr

def TimeStamp(ts = None):
	"""
	Create a nicely formatted string timestamp from an existing date+time or
	just from now() if none is provided.
	---
	Usage: timestamp = TimeStampStr(ts)
	---	
	ts: datetime.datetime timestamp; if None*, this will use datetime.datetime.now()
	timestamp: nicely formatted timestamp in format YYYYMMDD_HHMMSS
	---
	ex: print(QB.TimeStamp)
	JAH 20121031
	"""	
	
	if (ts is None) or (type(ts) is not dat.datetime):
		ts = dat.datetime.now()
	
	# JAH 20140108 changed to accomodate python 3
	#return string.replace(string.replace(string.replace(('%s'%ts)[:19],' ','_'),'-',''),':','')
	return ('%s'%ts)[:19].replace(' ','_').replace('-','').replace(':','')

def SignCumCount(data):
	"""	
	Compute a cumulative count of the number of days in the same direction for a set of 
	p variables. Because this does a 1-day diff, the 1st element will be 0.
	---
	Usage: counts = SignCumCount(data)
	---
	data: (n,p) data array_like of n observations of p variables (or (n,) vector)
	counts: array same size as data, holding the in-sign counts; first row is 0s
	---
	ex: 
	JAH 20121206
	"""
	
	# force data to be a 2d matrix, even if there is only 1 variables
	data = np.array(data, copy = False)				# first force to an array
	oldshp = data.shape									# now get the old shape
	data = np.array(data, ndmin = 2, copy = False)	# force to 2d
	if data.shape[0] == 1:								# if 1d forced to 2d, will be (1,n), so transpose
		data = data.T									# this is ok to do, since the reshape will force a copy
	(n,p) = data.shape	
	
	# get the sign of each daily change
	signs = np.sign(Diff(data,1))
	
	# fill in the in-sign counts
	counts = signs.copy()
	# haaate looping, but can't figure out more efficient way to do this...
	for dim in range(p):
		tmp = signs[1,dim]
		for obs in range(2,n):
			if counts[obs,dim] == tmp:	# if same, increment and store
				counts[obs,dim] += counts[obs-1,dim]
			else:
				tmp = counts[obs,dim]	# not same, so reverse
	
	return np.reshape(counts,oldshp)

def Check_Holes(ticker, from_date=dat.date(1000,1,1), to_date=dat.date.today(),plot=False):
	"""
	Find and list days for which a ticker should have a price (non-holiday) but
	doesn't. Maybe plot it.
	---
	Usage: dates = Check_Holes(ticker, from_date, to_date, plot)
	---
	ticker: string indicating ticker to extract
	from_date* / to_date*: datetime date variables date range
	plot*: True = plot ticker in date range showing missing days; False* = don't
	dates: array holding missing dates (as datetime.date)
	---
	ex: d = QB.Check_Holes(ticker="SPX", from_date=dat.date(2012,9,1), to_date=dat.date(2012,12,31),plot=True)
	JAH 20130321
	"""
 
	# duck type ticker, maybe it's np.string, by trying to uppercase it
	# this won't work if it's not a string-type variable JAH 20130320
	try:
		ticker = ticker.upper()
	except AttributeError:
		raise TypeError("Ticker must be a string: %s"%Check_Holes.__doc__)

	# ensure input variables are correct
	if (type(from_date) is not dat.date) or (type(to_date) is not dat.date):
		raise TypeError("Variable(s) are wrong type: %s"%Check_Holes.__doc__)
	
	# 1) get calendar in date range
	dates, holidays = QD.GetTickerCalendar(ticker, from_date, to_date, trim=True)
	# 2) get all prices
	d,h,p = QD.GetTickerPrices_L(ticker,dates)
	# 3) get list of 0 days
	hole_days = d[(p==0).flatten() & ~(h)]
	# 4) plot (maybe)
	if plot:
		junk,tickdata = QD.GetTickers(filt_tick = "='%s'"%ticker)
		tit = 'Missing Days Plot(%d)\n%s: %s'%(len(hole_days),ticker,tickdata[0][1])
		fh = plt.figure()
		PlotTickers(tit,ticker,p,d,h,fill=True,ax=plt.subplot(1,1,1))
		plt.grid('on')
		# annotate holes - have to loop :-(
		for hole in hole_days:
			thisday = np.where(d == hole)[0]
			plt.annotate('*'+hole.strftime('%Y-%b-%d'),(thisday,p[thisday]),\
			rotation=45,color='r',rotation_mode='anchor')

	return hole_days

def Check_Spikes(ticker, from_date=dat.date(1000,1,1), to_date=dat.date.today(),plot=False):
	"""
	Find and list days for which a ticker's price changes by "too much".  This is
	determined by taking the median daily price change and a robust estimate of
	the standard deviation, using IQR/1.348, and computing a range of median +/1 6sd.
	Any price changes bigger than this are flagged as spikes.
	---
	Usage: dates = Check_Spikes(ticker, from_date, to_date, plot)
	---
	ticker: string indicating ticker to extract
	from_date* / to_date*: datetime date variables date range
	plot*: True = plot ticker in date range showing spike days; False* = don't
	dates: array holding spike dates (as datetime.date)
	---
	ex: d = QB.Check_Spikes(ticker="XAX", from_date=dat.date(2007,1,1), to_date=dat.date(2007,6,30),plot=True)
	JAH 20130325
	"""
 
	# duck type ticker, maybe it's np.string, by trying to uppercase it
	# this won't work if it's not a string-type variable JAH 20130320
	try:
		ticker = ticker.upper()
	except AttributeError:
		raise TypeError("Ticker must be a string: %s"%Check_Spikes.__doc__)

	# ensure input variables are correct
	if (type(from_date) is not dat.date) or (type(to_date) is not dat.date):
		raise TypeError("Variable(s) are wrong type: %s"%Check_Spikes.__doc__)
	
	# 1) get calendar in date range
	dates, holidays = QD.GetTickerCalendar(ticker, from_date, to_date, trim=True)
	# 2) get all prices and remove holidays
	d,h,p = QD.GetTickerPrices_L(ticker,dates)
	d = d[~h]
	# 3) get the % daily changes
	p = FillHoles(p[~h].flatten())[0]
	dailyrets = Diff(p,1) / LagLead(p,1)
	# 4) identify potential spikes
	ret_median = sstat.scoreatpercentile(dailyrets,50)
	ret_IQR = sstat.scoreatpercentile(dailyrets,75)-sstat.scoreatpercentile(dailyrets,25)
	# get robust standard deviation estimate
	ret_sd = ret_IQR/1.348
	ret_robust_range = [ret_median - 6*ret_sd, ret_median +  6*ret_sd]
	# see which days are outside of the +/- 6sigma range
	ret_big = (dailyrets < ret_robust_range[0]) | (dailyrets > ret_robust_range[-1])
	# now correct this to remove the "correction" of the spike
	ret_big_L = LagLead(ret_big,1)
	ret_big = (ret_big_L!=ret_big) & (ret_big==True)
	# list of spike days
	spike_days = d[ret_big]
	# 5) plot (maybe)
	if plot:
		junk,tickdata = QD.GetTickers(filt_tick = "='%s'"%ticker)
		fh = plt.figure()
		ax1 = plt.subplot(2,1,1)
		tit = 'Spikes Plot(%d)\n%s: %s'%(len(spike_days),ticker,tickdata[0][1])
		PlotTickers(tit,ticker,p[1:],d[1:],None,fill=True,ax=ax1)
		plt.grid('on')
		ax2 = plt.subplot(2,1,2)
		PlotTickers('Daily Returns','Daily Returns',dailyrets[1:],d[1:],None,fill=False,ax=ax2)
		plt.grid('on')
		# annotate spikes - have to loop :-(
		for spike in spike_days:
			thisday = np.where(d == spike)[0]
			ax1.annotate('*'+spike.strftime('%Y-%b-%d'),(thisday,p[thisday]),\
			rotation=45,color='r',rotation_mode='anchor')
			ax2.annotate('*'+spike.strftime('%Y-%b-%d'),(thisday,dailyrets[thisday]),\
			rotation=45,color='r',rotation_mode='anchor')	
		
	return spike_days

def DailyValidate(ticker, to_date=dat.date.today()+dat.timedelta(days=-1), lookback = 60,plot=False):
	"""
	Check the last n trading days up to a specified end date for holes & spikes.
	The checks use the same logic as Check_Holes & Check_Spikes. Two arrays of
	dates are output. This is designed to be run daily after data is loaded.
	---
	Usage: miss_dates, spike_dates = DailyValidate(ticker, to_date, lookback, plot)
	---
	ticker: string indicating ticker to validate
	to_date*: datetime date variables date range; default is computed yesterday
	lookback*: integer number of days to look back; default is 60
	plot*: True = create a 3-row subplot for missing days, spike days, and daily
		returns ONLY IF there are some missing/spikes; False* = don't
	miss_dates: array holding missing dates (as datetime.date), or None
	spike_dates: array holding spike dates (as datetime.date), or None
	---
	ex: md,sd = QB.DailyValidate(ticker="SPX",to_date = dat.date(2012,12,31), lookback = 60, True)
	JAH 20130409
	"""	
 
	# duck type ticker, maybe it's np.string, by trying to uppercase it
	# this won't work if it's not a string-type variable
	try:
		ticker = ticker.upper()
	except AttributeError:
		raise TypeError("Ticker must be a string: %s"%DailyValidate.__doc__)
	
	# ensure input variables are correct
	if (type(lookback) is not int) or (type(to_date) is not dat.date):
		raise TypeError("Variable(s) are wrong type: %s"%DailyValidate.__doc__)	
	
	# compute the from_date first
	tmp = to_date + dat.timedelta(days=-2*lookback)
	try:
		d,h = QD.GetTickerCalendar(ticker,tmp,to_date,True)
	except ValueError:
		# if GetTickerCalendar returns ValueError, it means no data in those days
		print('!!!No data for %s between %s and %s!!!'%(ticker,tmp,to_date))
		# so just return as missing the entire ticker calendar in the date range
		d,h = QD.GetTickerCalendar(ticker,tmp,to_date,False)
		return d[~h], np.array([])
		
	# keep an extra day on the from_date since we will compute the daily return for spikes check
	from_date = d[~h][-min(np.sum(~h),lookback)]
	
	# now get the trading calendar dates in [from_date-1, to_date]
	dates, holidays = QD.GetTickerCalendar(ticker, from_date, to_date, trim=True)
	d,h,p = QD.GetTickerPrices_L(ticker,dates)
	d = d[~h]; p = p[~h]
	
	# now find missing days, dropping the 1st, since it's before the lookup days range
	hole_days = d[1:][(p[1:]==0).flatten()]
	
	# get the % daily changes
	p = FillHoles(p.flatten())[0]
	dailyrets = Diff(p,1) / LagLead(p,1)
	# 4) identify potential spikes
	ret_median = sstat.scoreatpercentile(dailyrets,50)
	ret_IQR = sstat.scoreatpercentile(dailyrets,75)-sstat.scoreatpercentile(dailyrets,25)
	# get robust standard deviation estimate
	ret_sd = ret_IQR/1.348
	ret_robust_range = [ret_median - 6*ret_sd, ret_median +  6*ret_sd]
	# see which days are outside of the +/- 6sigma range
	ret_big = (dailyrets < ret_robust_range[0]) | (dailyrets > ret_robust_range[-1])
	# now correct this to remove the "correction" of the spike
	ret_big_L = LagLead(ret_big,1)
	ret_big = (ret_big_L!=ret_big) & (ret_big==True)
	# list of spike days
	spike_days = d[ret_big]
	
	# maybe plot - JAH added 20130526
	if plot and (len(hole_days)>0 or len(spike_days)>0):
		tickdata = QD.GetTickers(filt_tick = "='%s'"%ticker)[1]
		fh = plt.figure(figsize=(23,12))
		# MISSING DAYS		
		tit = 'Missing Days Plot(%d)\n%s: %s'%(len(hole_days),ticker,tickdata[0][1])
		ax0 = plt.subplot(3,1,1)
		PlotTickers(tit,ticker,p[1:],d[1:],None,fill=True,ax=ax0)
		plt.grid('on')
		# annotate holes - have to loop :-(
		for hole in hole_days:
			thisday = np.where(d == hole)[0]-1
			ax0.annotate('*'+hole.strftime('%Y-%b-%d'),(thisday-1,p[thisday]),\
			rotation=45,color='r',rotation_mode='anchor')
		# SPIKE DAYS
		tit = 'Spikes Plot(%d)\n%s: %s'%(len(spike_days),ticker,tickdata[0][1])
		ax1 = plt.subplot(3,1,2)
		PlotTickers(tit,ticker,p[1:],d[1:],None,fill=True,ax = ax1)
		plt.grid('on')
		ax2 = plt.subplot(3,1,3)
		PlotTickers('Daily Returns','Daily Returns',dailyrets[1:],d[1:],None,fill=False,ax=ax2)
		plt.grid('on')
		# annotate spikes - have to loop :-(
		for spike in spike_days:
			thisday = np.where(d == spike)[0]
			ax1.annotate('*'+spike.strftime('%Y-%b-%d'),(thisday-1,p[thisday]),\
			rotation=45,color='r',rotation_mode='anchor')
			ax2.annotate('*'+spike.strftime('%Y-%b-%d'),(thisday-1,dailyrets[thisday]),\
			rotation=45,color='r',rotation_mode='anchor')
	
	return hole_days, spike_days

def DailyValidateList(tickers='ALL', to_date=dat.date.today()+dat.timedelta(days=-1), lookback = 60, plot = False, save_path=None):
	"""
	Check the last n trading days of a list of tickers up to a specified end date
	for holes & spikes.  You can use the first output from QD.GetTickers to
	create a target list.  On that list, this merely calls DailyValidate. Returns
	four variables holding the hole / spike days and day counts.
	---
	Usage: holes, spikes = DailyValidateList(tickers, to_date, lookback, plot, save_path)
	---
	tickers: array_like of tickers to validate, if "ALL" will take all
	to_date*: datetime date variables date range; default is computed yesterday
	lookback*: integer number of days to look back; default is 60
	plot*: True = plot ticker in date range showing missing days; False* = don't
	save_path*: if plot is true and this is not None*, save & close the plots to ticker.png files
	holes: list holding for each ticker with >0 holes: ticker, count, array of dates
	spikes: list holding for each ticker with >0 spikes: ticker, count, array of dates
	---
	ex: ticks = QD.GetTickers(filt_type = "like '%MAJOR;%'")[0]
	holes, spikes = QB.DailyValidateList(ticks,to_date = dat.date(2012,12,31), lookback = 60, True)
	JAH 20130410
	"""	
	
	# ensure input variables are correct
	if (type(lookback) is not int) or (type(to_date) is not dat.date):
		raise TypeError("Variable(s) are wrong type: %s"%DailyValidateList.__doc__)	
	
	if (type(save_path) is not str) and (save_path is not None):
		raise TypeError('Variable save_path must be None or string path/filename: %s'%DailyValidateList.__doc__)
	
	# fix the save path if necessary - JAH 20130526
	if plot and (save_path is not None):
		save_path = save_path+'/'*(save_path[-1] != '/')
		if not os.path.exists(save_path):
			raise IOError('%s output path does not exist: %s'%(save_path,DailyValidateList.__doc__))
	
	# get ticker list maybe
	if type(tickers)==str and (tickers == 'ALL'):
		tickers,jnk = QD.GetTickers(filt_tick = "<>'_CHEAT'")
	elif type(tickers) == list:
		tickers = np.asarray(tickers)
	
	# setup output variables
	hole_counts = np.zeros(len(tickers),dtype=int)
	holes = []
	spike_counts = hole_counts.copy()
	spikes = []

	# loop through and validate them all
	for cnt in range(len(tickers)):
		hole,spik = DailyValidate(tickers[cnt],to_date,lookback,plot)
		holes.append(hole)
		hole_counts[cnt] = len(hole)
		spikes.append(spik)
		spike_counts[cnt] = len(spik)
		# if a plot was created, save it if appropriate - JAH 20130526
		if plot and (save_path is not None) and (hole_counts[cnt]>0 or spike_counts[cnt]>0):
			print('Saving plot (%d of %d): %s%s.png'%(cnt,len(tickers),save_path,tickers[cnt]))
			plt.savefig(save_path+tickers[cnt]+'.png')
			plt.close(plt.gcf())
			time.sleep(0.15)

	# prepare the indices for the final output arrays
	hole = hole_counts > 0
	spik = spike_counts > 0
	holes_out = [[tick,cunt,dats] for tick,cunt,dats in zip(tickers[hole], hole_counts[hole], np.asarray(holes)[hole])]
	spike_out = [[tick,cunt,dats] for tick,cunt,dats in zip(tickers[spik],spike_counts[spik],np.asarray(spikes)[spik])]

	return holes_out,spike_out

def QuickPlot(ticker, from_date=dat.date(1000,1,1), to_date=dat.date.today(), fill=True, ax=None):
	"""
	Plot a single ticker for a specified trading range.  This is basically a wrapper
	for PlotTickers, setting up the index name, dates, and prices.
	---
	Usage: fig = QuickPlot(ticker, from_date, to_date, fill, ax)
	---
	ticker: string ticker name
	from_date* / to_date*: datetime date variables date range
	fill*: default is True*; passed directly to PlotTickers
	ax*: default is None*; passed directly to  PlotTickers
	fig: figure handle
	---
	ex: QB.QuickPlot('USDEUR',from_date=dat.date(2013,1,1),fill=True)
	JAH 20130501
	"""
	
	# duck type ticker, maybe it's np.string, by trying to uppercase it
	# this won't work if it's not a string-type variable
	try:
		ticker = ticker.upper()
	except AttributeError:
		raise TypeError("Ticker must be a string: %s"%QuickPlot.__doc__)	
	
	# ensure input variables are correct
	if (type(from_date) is not dat.date) or (type(to_date) is not dat.date):
		raise TypeError("Dates should be dat.date: %s"%QuickPlot.__doc__)	
	

	# get the ticker name
	nam = QD.GetTickers(filt_tick ="='%s'"%ticker)[1][0][1]
	
	# get the dates
	d,h = QD.GetTickerCalendar(ticker,from_date,to_date,trim=True)
	d,h,p = QD.GetTickerPrices_R(ticker,d[0],d[-1])
	
	# now plot
	return PlotTickers(nam,ticker,p,d,h,fill,ax)

def PlotwDailyRet(ticker, from_date=dat.date(1000,1,1), to_date=dat.date.today(), fill=True):
	"""
	Plot a single ticker and it's daily price returns for a specified trading
	range.  This is basically a wrapper for PlotTickers.
	---
	Usage: fig = PlotwDailyRet(ticker, from_date, to_date, fill)
	---
	ticker: string indicating ticker to extract
	from_date* / to_date*: datetime date variables date range
	fill*: default is True*; passed directly to PlotTickers
	fig: figure handle
	---
	ex: QB.PlotwDailyRet('USDEUR',from_date=dat.date(2013,1,1),fill=True)
	JAH 20140201
	"""
 
	# duck type ticker, maybe it's np.string, by trying to uppercase it
	# this won't work if it's not a string-type variable JAH 20130320
	try:
		ticker = ticker.upper()
	except AttributeError:
		raise TypeError("Ticker must be a string: %s"%PlotwDailyRet.__doc__)

	# ensure input variables are correct
	if (type(from_date) is not dat.date) or (type(to_date) is not dat.date):
		raise TypeError("Variable(s) are wrong type: %s"%PlotwDailyRet.__doc__)
	
	# 1) get calendar in date range
	dates, holidays = QD.GetTickerCalendar(ticker, from_date, to_date, trim=True)
	# 2) get all prices and remove holidays
	d,h,p = QD.GetTickerPrices_L(ticker,dates)
	d = d[~h]
	# 3) get the % daily changes
	if fill:
		p = FillHoles(p[~h].flatten())[0]
	else:
		p = p[~h].flatten()
	dailyrets = Diff(p,1) / LagLead(p,1)
	# 4) plot
	# get the ticker name
	nam = QD.GetTickers(filt_tick ="='%s'"%ticker)[1][0][1]
	fh = plt.figure()
	ax1 = plt.subplot(2,1,1)
	PlotTickers(nam,ticker,p[1:],d[1:],None,fill=False,ax=ax1)
	plt.grid('on')
	ax2 = plt.subplot(2,1,2)
	PlotTickers('Daily Returns','Daily Returns',dailyrets[1:],d[1:],None,fill=False,ax=ax2)
	plt.grid('on')
		
	return fh
