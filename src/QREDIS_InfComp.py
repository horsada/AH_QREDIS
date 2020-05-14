#QREDIS_InfComp
"""
QREDIS information complexity functions
----------------------
ComputeIFIM - 20120920 - compute the inverse Fisher information matrix for a multivariate Gaussian model
DupMatrix - 20120920 - generate the duplication matrix & its Moore-Penrose inverse
EntComp - 20120919 - compute the entropic complexity of a covariance matrix
RegressIC - 20121101 - fit a multiple regression model, then score information criteria
----------------------
JAH 20140117 everything has been tested and seems to be working fine with python 3
"""

import string
import math
import numpy as np
from numpy import linalg
import QREDIS_Basic as QB

def EntComp(S, calctype):
	"""
	Compute the C1 or C1F (Frobenius norm) entropic complexity of a covariance matrix.
	---
	Usage: comp = EntComp(S, calctype)
	---
	S: (p,p) array_like covariance matrix of data (calc type = 0) or IFIM (calc type = 1,-1)
	calctype: 1 = compute C1 complexity of matrix; 0 = compute complexity of IFIM
		depending on sigma and Dp by opening it up; -1 = compute C1F
	comp: scalar comlexity of covariance
	---    
	ex: n = 20; S = np.cov(rnd.rand(n,1).T); C11 = QI.EntComp(ComputeIFIM(S,n),1); C10 = QI.EntComp(S,0); print(C11); print(C10)
	JAH 20120919
	"""

	# duck covariance matrix into a 2d matrix
	S = np.array(S, ndmin = 2, copy=False)
	
	# get proper dimensions	
	(p,p2) = np.shape(S)

	# finally check this is a square matrix, can be univariate standard deviation IF opened up calc
	if (p != p2) or ((p == 1) and (calctype != 0)) or (S.ndim > 2):
		raise ValueError("Covariance matrix should be an array of size (p,p): %s"%EntComp.__doc__)
	if (abs(calctype) > 1) or (type(calctype) is not int):
		raise ValueError("Calculation must be an integer of either -1, 0, or 1: %s"%EntComp.__doc__)

	# compute some things only once
	cmdet = linalg.det(S)
	cmtra = np.trace(S)
	cmdim = linalg.matrix_rank(S)

	# check for bad determinant
	if (cmdet == 0) and (calctype != -1):
		print("WARNING: Matrix determinant is 0!")
		return -np.inf
	elif cmdet < 0:
		print("WARNING: Matrix determinant is <0!")
		return -np.inf

	# compute the complexity - in all this I use np.log because it returns -inf for det = 0
	if calctype == 1:
		# comp = (rank/2) * np.log(trace/rank) - np.log(determ)/2
		C1 = cmdim*0.5*np.log(cmtra/cmdim) - 0.5*np.log(cmdet)
	elif calctype == 0:
	# "opened up"; input is Sigma
		p1 = (2*p + p*(p+1))/4.0
		p2 = cmtra + 0.5*np.trace(linalg.matrix_power(S,2)) + \
			0.5*cmtra**2 + np.sum(np.diag(S)**2)
		p3 = 0.5*(2*p + p*(p+1))
		p4 = 0.5*(np.log(cmdet)*(p+2) + p*np.log(2))
		C1 = p1*np.log(p2/p3) - p4
	elif calctype == -1:
		# frobenius norm version of complexity                
		(eigval,mn) = linalg.eig(S)		# mn really holds eigvecs here, but will reuse
		mn = np.mean(eigval)
		C1 = np.sum((eigval - mn)**2) / (4.0*mn**2)

	return C1
	
def DupMatrix(p):
	"""
	Returns Magnus and Neudecker's duplication matrix of size p, and the moore-penrose
	inverse of Dp, noted as Dp+.
	---
	Usage: dup. matrix, moore-pen. inverse = DupMatrix(p)
	---
	p: integer number dimensions required
	dup. matrix: (p**2, p*(p+1)/2) duplication matrix
	moore-pen. inverse: Moore-Penrose inverse of duplication matrix
	---    
	ex: Dp,mopeninv = QI.DupMatrix(3)
	JAH 20120920
	"""
	
	# ensure proper arguments
	if (type(p) is not int) or (p < 1):
		raise ValueError("Number dimensions p must be integer >= 1: %s"%DupMatrix.__doc__)

	# algorithm from COMPLETELY UNDOCUMENTED Prof. Hamparsum Bozdogan code
	
	# first prepare the indices of where we will put a 1
	# fill in a lower triangular matrix with numbers from 1:p*(p+1)/2...
	# but should count DOWN columns instead of the natural across rows...
	# so have to use an upper triangular than transpose
	myones = np.triu(np.ones((p,p)))
	myones[myones !=0] = np.array(range(int(p*(p+1)/2)))
	myones = myones.T
	# now we take what's strictly below the diag and mirror it above
	myones = myones + np.tril(myones,-1).T
	# finally flatten it
	myones = myones.flatten()

	# prepare the dup matrix
	Dp = np.zeros((p*p,p*(p+1)/2))
	# and fill in the 1s	
	for i in range(p*p):
		Dp[i,myones[i]] = 1
	
	# finally compute the moore-penrose inverse
	mopeninv =  np.dot(linalg.inv(np.dot(Dp.T,Dp)),Dp.T)
	
	return Dp,mopeninv;
	
def ComputeIFIM(S,n):
	"""
	This function will compute the full inverse Fisher information matrix for a
	multivariate gaussian model.
	---
	Usage: IFIM = ComputeIFIM(S, n)
	---
	S: (p,p) array_like estimated covariance matrix
	n: integer number observations, used to scale IFIM by 1/n
	IFIM: p*(p+3)/2 square inverse Fisher information matrix
	---    
	ex: n = 20; S = np.cov(rnd.rand(n,1).T); IFIM = QI.ComputeIFIM(S,n); print(IFIM)
	JAH 20120920
	"""
	
	# duck covariance matrix into a 2d matrix
	S = np.array(S, ndmin = 2, copy=False)
	(p,p2) = np.shape(S)		# define dimensions p, and p2	
	
	# ensure proper arguments
	if (p != p2) or (S.ndim > 2):
		raise ValueError("Input matrix must either be (n,p) or (p,p) array: %s"%ComputeIFIM.__doc__)
	if type(n) is not int:
		raise ValueError("Number observations must be integer: %s"%ComputeIFIM.__doc__)

	# compute quadrants
	IFIMp = p*(p + 3)/2					# dimension of IFIM
	dupmat,mopeninv = DupMatrix(p)		# compute the duplication matrix and its moore-penrose inverse
	Q1 = np.zeros((p,IFIMp - p))		# upper-right and lower-left (transposed) quadrant
	Q4 = 2*np.dot(np.dot(mopeninv, np.kron(S,S)),mopeninv.T)	#lower-right quadrant
	
	# put it all together now
	return np.vstack((np.hstack((S,Q1)), np.hstack((Q1.T,Q4))))/n;

def RegressIC(IC,X, Y, rtype, ridgeparm=None):
	"""	
	Perform the required multiple regression using RegressMe in QREDIS_Basic, then
	score said model using the desired information criteria, or just the maximized
	likelihood (LL). An intercept is always included in the model and calculation.
	---
	Usage: ICscore, regress_results = RegressIC(IC,X, Y, rtype, ridgeparm)
	---
	IC: information criteria to use; choices are LL, AIC, SBC, CAIC, ICOMP,
		ICOMP_PEU, ICOMP_PEULN
	X: (n,p) array_like of independent data
	Y: array_like of size n of dependent data
	rtype: type of regression (see RegressMe)
	ridgeparm*: parameter(s) for ridge regression (see RegressMe)
	ICscore: float value of the selected criterion for the selected regression model
	regress_results: tuple holding RegressMe output (coefficients, statistics, 
		inverse design matrix)
	---    
	ex: 
	JAH 20121101
	"""
	
	# check IC code; no checking of parameters for RegressMe - let that happen there
	try:
		IC = IC.upper()
	except AttributeError:
		raise TypeError("IC must be a string: %s"%RegressIC.__doc__)
		
	ICs = ['LL','AIC','SBC','CAIC','ICOMP','ICOMP_PEU','ICOMP_PEULN']
	if (type(IC) is not str) or (IC not in ICs):
		raise ValueError('IC must be in %s: %s'%(ICs,RegressIC.__doc__))
	
	# run the regression, just passing through parameters
	betas,stats,invdes = QB.RegressMe(X, Y, rtype, ridgeparm,-1)
	
	# check for infinite output from regressme and exit with score = Inf
	if not(np.isfinite(stats[0])):
		return np.inf,(betas,stats,invdes)
	
	# parse some stats from RegressMe	
	n = float(stats[3]+1)			# SST dof is n-1
	mse = stats[6]/n				# mean squared error
	q = stats[5] + 1				# q is number coefficients; comes from: SSR dof is p-1 where p is size(X+1s)[1]
	numparmsest = q + 1				# parms est = # beta coefs + error variance
	
	# now compute the selected information criteria
	lackoffit = n*math.log(2*math.pi) + n*math.log(mse) + n
	if IC == 'LL':							# just the maximized likelihood
		lackoffit = -0.5*lackoffit
		penalty = 0
	if IC == 'AIC':							# 2*(# parms)
		penalty = 2*numparmsest
	elif IC == 'SBC':						# log(n)*(# parms)
		penalty = math.log(n)*numparmsest
	elif IC == 'CAIC':						# (log(n)+1)*(# parms)
		penalty =  (math.log(n) + 1)*numparmsest;
	elif IC in ['ICOMP','ICOMP_PEU','ICOMP_PEULN']:
		# first build the IFIM
		zmat = np.zeros(q).reshape((q,1))
		IFIM = np.vstack((np.hstack((mse*invdes,zmat)),np.append(zmat.T,2*mse**2/n)))
		# now comupute the base ICOMP penalty
		penalty = 2*EntComp(IFIM,1);		# 2*C1(Finv)
		# and now the addons for other ICOMPs
		if IC == 'ICOMP_PEU':				# # parms + 2*C1(Finv)
			penalty = numparmsest + penalty
		elif IC == 'ICOMP_PEULN':			# # parms + log(n)*C1(Finv)
			penalty = numparmsest + math.log(n)*penalty/2.0

	return lackoffit + penalty, (betas, stats, invdes)
