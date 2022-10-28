#QMod_RegressGASubPred
"""
QREDIS Regression with GA Subsetting Prediction model
"""

class QMod_RegressGASubPred(QMod_Template):
	"""
	General Methods for External Use; THESE SHOULD NOT BE OVERRODE IN SPECIFIC MODEL CLASSES:
	Describe - Save a description in the QREDIS database
	ClearParams - Clear parameters from the QREDIS database for specified dates
	ClearPS - Clear parameters and signals from the QREDIS database for specified dates
	ClearSignals - Clear signals from the QREDIS database for specified dates
	GetSettings - Get model settings from the QREDIS database
	GetSignals - Get model signals from the QREDIS database for a specified date
	NextTradeBlock - Calculate the block of dates for the next trade
	NextTrainTradeBlock - Calculate the date blocks for the next model train
	RetrainCheck - Check if the model needs to be retrained
	----------------------
	Model-Specific Methods; THESE MUST BE OVERRODE IN SPECIFIC MODEL CLASSES:
	_ParseParams - Read parameters from the QREDIS database into variables
	_ParseSettings - Read model settings from the QREDIS database into variables
	_ParseSource - Parse the source_data string from the QREDIS database into a list
	_PrintSettings - Make a nice printable string with the model settings (for print model)
	_SetMe - Save default model settings to the QREDIS database; called when model is initialized
	GetParams - Get parameters from the QREDIS database for a specified date
	Train - Train the model for a specific period and save parameters (and other info) to the QREDIS database
	Trade - Trade the model for a specific day and save the signal to the QREDIS database
	----------------------	
	This model trades the target index by selecting, using the GA & information	criteria,
	a subset regression model to predict the daily returns. X is composed of lags
	1 - numlags of daily returns of the source_data. Model settings include the max number
	lags, information criteria used by the GA, regression type, ridge param if necessary,
	signal generation threshhold, and whether or not the training data is attenuated.
	The settings dict should have elements:
		'numlag' (int), example = 5
		'IC' (str), example = 'SBC'
		'regtype' (str), example = 'R'
		'ridge' (float / None), example = 0.001
		'sigthresh' (float), example = 0.001
		'atten' (bool), example = True
	Load using exec(open(QREDIS_mod+'/QMod_RegressGASubPred.py').read())
	JAH 20121103
	"""

########################## MODEL MAKER SHOULD EDIT THESE! ###############################

	def _SetMe(self, mysettings):
		# THIS SHOULD ONLY BE RUN AS PART OF THE FIRST MODEL INITIALIZATION
		# DURING MODELING, USE _ParseSettings TO SET THEM FROM THE DB!!!
		# define and save model settings (settings as opposed to parameters, which can
		# change over time; settings are constant
		self.numlags = mysettings['numlags']		# lag source returns 1-day, 2-days, ... up to this
		self.IC = mysettings['IC']					# information criteria used
		self.regtype = mysettings['regtype']		# type of regression
		self.ridge = mysettings['ridge']			# ridge parameter if regtype=R
		self.sigthresh = mysettings['sigthresh']	# threshhold for converting yhat to signal
		self.atten = mysettings['atten']			# should history be attenuated in training?
		# now save
		self._SaveSettings(['numlags','IC','regtype','ridge','sigthresh','atten'],\
			[self.numlags,self.IC,self.regtype,self.ridge,self.sigthresh,self.atten],\
			['int','str','str','float','float','bool'])
		return True
	
	def _PrintSettings(self):
		# this prints the model settings; it is called from the __str__ function
		return '\tMax Lags: %d\n\tInf. Criteria: %s\n\tRegression Type: %s(%s)\n\tSignal Threshhold: %0.6f\n\tAttenuate: %s'\
			%(self.numlags, self.IC, self.regtype, self.ridge, self.sigthresh, self.atten)
		
	def _ParseSettings(self):
		# get the saved settings
		nams,vals,typs = self.GetSettings()
		for n,v,t in zip(nams, vals, typs):
			if v == 'None':
				exec("self."+n+" = None")			# bypass type if None was stored
			else:
				exec("self.%s = %s('%s')"%(n,t,v))

	def _ParseParams(self, param_date):
		# extract the parameters from the db
		nams, vals, typs = QD.GetParams(self.model_id, param_date)
		# since GetParameters extracts everything from the database as a pair of
		# arrays of strings, here we parse them, using the stored format/type
		for n,v,t in zip(nams, vals, typs):
			if v == 'None':
				exec("self."+n+" = None")			# bypass type if None was stored
			elif t == 'list':
				exec("self."+n+" = eval(v)")		# list, so use eval to rebuild it
				if n == "betas":					# make betas into 2-d column array
					self.betas = np.array(self.betas,ndmin=2,dtype=float).T
			else:
				exec("self.%s = %s('%s')"%(n,t,v))

	def GetParams(self,param_date):
		"""
		Extract stored model parameters from the QREDIS Database for a specified
		date, which is required. Returns a dictionary holding the parameters.
		JAH 20121128
		"""
		self._ParseParams(param_date)
		return {'walk_randstate':self.walk_randstate,'randseed':self.randseed,'missd_days':self.missd_days,\
			'GAscore':self.GAscore,'intercept':self.intercept,'betas':self.betas,'inds':self.inds,'lags':self.lags}
		
	def _ParseSource(self):
		"""
		Parse the source_date string as extracted from a database.  It should be a
		a) list of tickers, b) filt_type string, or c) list of tickers with the first
		item being a filt_type string. If it is mixed (like c), the 2 sections should
		be separated by a ":".
		JAH 20130526
		"""
		if type(self.source_data) is str:
			# check if 1st item is a filt_type filter
			tmp = self.source_data.split(":")
			if (tmp[0][:4] == "like") or (tmp[0][:1]=="="):
				# get the tickers from the filt_type
				self.source_data = QD.GetTickers(filt_type = tmp[0])[0].tolist()
			else:
				# first item is not filt_type, so all of source_data is just a list
				self.source_data = eval(tmp[0])
			if len(tmp)>1:
				# there is a second item after split, so must be a list
				self.source_data += eval(tmp[1])

	def Train(self,train_dates,trade_dates,randstate=None):
		# maybe there are holes on the first day wich will cause an annoying (but unimportant) error
		# in the divsion, turn off the error
		oldsett = np.seterr(divide='ignore')
		
		# get and prepare the source data; dont' care about dates & holidays here because
		# the Train function should be called already with the dates correct for the target index
		self._ParseSource()	# JAH added 20130526
		d, h, prices = QD.GetTickersPrices_L(self.source_data, train_dates)
		# maybe there are some holes here, since maybe target_index is on a different calendar, so fill
		prices, filld = QB.FillHoles(prices)
		# get the daily returns - can't forget to delete 1st row of nans later!
		dailyrets = QB.Diff(prices,1) / QB.LagLead(prices,1)
		# we will try # lags for each ticker, so duplicate
		dailyrets = np.repeat(dailyrets,self.numlags,axis=1)
		# get the 1-day - #-day lags, then cut off the buffer rows
		lags = list(range(1,self.numlags+1))*(prices.shape[1])
		X_sourceret = QB.LagLead(dailyrets,lags)[self.buffer_days:,:]
		
		# get and prepare the target data; train_dates should already have target holidays removed
		d, h, target = QD.GetTickerPrices_L(self.target_index, train_dates)
		# if missing data, fill holes (hopefully none)
		target,filld = QB.FillHoles(target)
		# maybe there are holes on the first day which will cause an annoying (but unimportant) error
		# in the divsion, so fill in nans, I guess
		target[0,target[0,:] == 0] = np.nan
		# now get the daily returns and keep only non-buffer days
		Y_targetret = (QB.Diff(target,1) / QB.LagLead(target,1))[self.buffer_days:,:]
		
		# finally, since we are using 60 training days, attenuate JAH 20121110
		if self.atten:
			X_sourceret = QB.Attenuate(X_sourceret)
			Y_targetret = QB.Attenuate(Y_targetret)
		
		# set up for the GA now
		GA_data = {'data':X_sourceret, 'data_name':'%s'%(self.name)}
		GA_objec_prms = {'function':'QI.RegressIC','israndom':False,\
			'data':'X','Y':Y_targetret,'rtype':self.regtype,'ridgeparm':self.ridge,'IC':self.IC}
		GA_parms = {'init_perc':0.15,'showtopsubs':5,'popul_size':100,'num_generns':100,'nochange_terminate':80,\
			'convgcrit':0.0001,'elitism':False,'mate_type':2,'prob_xover':0.75,'xover_type':1,\
			'prob_mutate':0.1,'prob_engineer':0.50,'optim_goal':-1,'plotflag':False,'printfreq':5,\
			'randstate':0,'seed_vars':[],'force_vars':[]}			

		# set the random generator state if input from WalkForward JAH 20121112
		# JAH 20121111 5 GA runs so 5 random generator seeds
		if randstate is not None:
			randseeds = [randstate, randstate+1, randstate+2, randstate+3, randstate+4]
		else:
			randseeds = [0,0,0,0,0]
		
		# run the GA 5 times and take the best 5 sols from each		
		sols = np.ones((25,len(self.source_data)*self.numlags+2),dtype=float)*np.inf
		# 1st
		if randstate is None:
			jnk = dat.datetime.now()
			randseeds[0] = jnk.hour*10000+jnk.minute*100+jnk.second
		GA_parms['randstate'] = randseeds[0]
		tmp  = QG.RunGA(GA_data, GA_objec_prms, GA_parms, None)[0]
		sols[0:5,:] = tmp[:5,:]
		#2nd
		if randstate is None:
			jnk = dat.datetime.now()
			randseeds[1] = jnk.hour*10000+jnk.minute*100+jnk.second
		GA_parms['randstate'] = randseeds[1]
		tmp = QG.RunGA(GA_data, GA_objec_prms, GA_parms, None)[0]
		sols[5:10,:] = tmp[:5,:]
		# 3rd
		if randstate is None:
			jnk = dat.datetime.now()
			randseeds[2] = jnk.hour*10000+jnk.minute*100+jnk.second
		GA_parms['randstate'] = randseeds[2]
		tmp = QG.RunGA(GA_data, GA_objec_prms, GA_parms, None)[0]
		sols[10:15,:] = tmp[:5,:]
		# 4th
		if randstate is None:
			jnk = dat.datetime.now()
			randseeds[3] = jnk.hour*10000+jnk.minute*100+jnk.second
		GA_parms['randstate'] = randseeds[3]
		tmp = QG.RunGA(GA_data, GA_objec_prms, GA_parms, None)[0]
		sols[15:20,:] = tmp[:5,:]
		# 5th
		if randstate is None:
			jnk = dat.datetime.now()
			randseeds[4] = jnk.hour*10000+jnk.minute*100+jnk.second
		GA_parms['randstate'] = randseeds[4]
		tmp = QG.RunGA(GA_data, GA_objec_prms, GA_parms, None)[0]
		sols[20:25,:] = tmp[:5,:]		
		# ... and get the overall best
		bst = np.argmax(sols[:,0]*GA_parms['optim_goal'])
		best_bin = sols[bst,2:] == 1
		rnd = randseeds[bst//5]	# this makes use of forced integer div which rounds down
		# JAH 20121221, in preparation for future use of 3.x, force integer division with //, since floor
		# division will no longer be the default

		# now run the regression last time to get the coefficients
		betas,stats,invdes = QB.RegressMe(X_sourceret[:,best_bin], Y_targetret, self.regtype, self.ridge)
		betasl = '%s'%['%0.5f'%b for b in betas[1:,0]] # make flat string list of coefs w/o intercept
		betasl = string.replace(betasl,"'","")
		# get the best indices & lags chosen & prepare the string lists for parameter saving
		sel_inds = list(np.repeat(self.source_data,self.numlags)[best_bin])
		sel_inds = ('%s'%(sel_inds)).replace("'",'"')
		sel_lags = list(np.array(lags)[best_bin])
		sel_lags = '%s'%(sel_lags)
		print('--- GA Selection: inds = %s, lags = %s, score = %0.4f ---'%(sel_inds,sel_lags,sols[bst,0]))
		
		# prepare parameters/results ...
		param_names = ['walk_randstate','randseed','missd_days','GAscore','intercept','betas','inds','lags']
		param_values = [randstate,rnd,filld,sols[bst,0],betas[0,0],betasl,sel_inds,sel_lags]
		param_types = ['int','int','int','float','float','list','list','list']
		# ... and then save then
		self._SaveParams(trade_dates[self.buffer_days:], param_names, param_values, param_types)
		# finally reset the numpy floating point error settings
		np.seterr(divide=oldsett['divide'])
		return True

	def Trade(self,buffertradedates):
		# get and parse parameters to use
		self._ParseParams(buffertradedates[-1])
		# get the source data
		self._ParseSource()	# JAH added 20130526
		d, h, p = QD.GetTickersPrices_L(self.source_data, buffertradedates)
		p,filld = QB.FillHoles(p)
		# get the correct specific indices
		prices = np.zeros((p.shape[0],len(self.inds)),dtype=float)
		for cnt in range(len(self.inds)):
			prices[:,cnt] = p[:,self.source_data.index(self.inds[cnt])]
		# compute the daily returns
		dailyrets = QB.Diff(prices,1) / QB.LagLead(prices,1)
		# lag each ticker appropriately, then take the last day
		X = QB.LagLead(dailyrets,self.lags)
		X = np.array(X[-1,:],ndmin=2)
		# compute the regression
		yhat = self.intercept + np.dot(X,self.betas)
		
		# use a threshold to convert signal, then save it
		self.last_signal = int(np.sign(yhat)*(abs(yhat) >= self.sigthresh))
		self._SaveSignal(buffertradedates[-1])
		return self.last_signal
