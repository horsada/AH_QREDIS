#QMod_PersRever_Select
"""
QREDIS Persistence Reverse Select Model
"""

class QMod_PersRever_Select(QMod_Template):
	"""
	General Methods for External Use; THESE SHOULD NOT BE OVERRODE IN SPECIFIC MODEL CLASSES:
	Describe - Save a description in the QREDIS database
	ClearParams - Clear parameters from the QREDIS database for specified dates
	ClearPS - Clear parameters and signals from the QREDIS database for specified dates
	ClearSignals - Clear signals from the QREDIS database for specified dates
	GetParams - Get parameters from the QREDIS database for a specified date
	GetSettings - Get model settings from the QREDIS database
	GetSignals - Get model signals from the QREDIS database for a specified date
	NextTradeBlock - Calculate the block of dates for the next trade
	NextTrainTradeBlock - Calculate the date blocks for the next model train
	RetrainCheck - Check if the model needs to be retrained
	----------------------
	Model-Specific Methods; THESE MUST BE OVERRODE IN SPECIFIC MODEL CLASSES:
	_ParseParams - Read parameters from the QREDIS database into variables
	_ParseSettings - Read model settings from the QREDIS database into variables
	_PrintSettings - Make a nice printable string with the model settings (for print model)
	_SetMe - Save default model settings to the QREDIS database; called when model is initialized
	Train - Train the model for a specific period and save parameters (and other info) to the QREDIS database
	Trade - Trade the model for a specific day and save the signal to the QREDIS database
	----------------------
	Evaluate a training period of a specific index (presumably the target index) and 
	count the # of days in each sign (up1, up2, dn1, up1, dn1, dn2, dn3, ...) based on
	# of occurences of each in-direction day, and the relative strength of the tendency
	to persist (1 -> 2 or -2 -> -3) or reverse (1 -> -1 or -2 -> 1), identify a subset
	of these tendencies to trade duing the trading period.  The # of occurences threshold
	is a % of the training period, and the tendency is measure as the ratio of the
	strongest tendency / weakest being larger than a specified value.
	The settings dict should have elements:
		'prratio_thresh' (float), example = 1.75
		'cumdays_thresh' (float), example = 0.1
	Load using exec(open(QREDIS_mod+'/QMod_PersRever_Select.py').read())
	"""
		
########################## MODEL MAKER SHOULD EDIT THESE! ###############################

	def _SetMe(self, mysettings):
		# THIS SHOULD ONLY BE RUN AS PART OF THE FIRST MODEL INITIALIZATION
		# DURING MODELING, USE _ParseSettings TO SET THEM FROM THE DB!!!
		# define and save model settings (settings as opposed to parameters, which can
		# change over time; settings are constant
		self.prratio_thresh = mysettings['prratio_thresh']		# threshold for a strong/weak disparity to be "significant"
		self.cumdays_thresh = mysettings['cumdays_thresh']		# necessary min observations (%) of a cum day count
		# now save		
		self._SaveSettings(['prratio_thresh','cumdays_thresh'], [self.prratio_thresh,self.cumdays_thresh], ['float','float'])
		return True
	
	def _PrintSettings(self):
		# this prints the model settings; it is called from the __str__ function
		return '\tThresholds=\n\tStrong/Weak Tendency Ratio: %0.4f\n\tCum Day Perc: %0.2f'%(self.prratio_thresh, self.cumdays_thresh)
		
	def _ParseSettings(self):
		# get the saved settings
		nams,vals,typs = self.GetSettings()		
		for n,v,t in zip(nams, vals, typs):
			exec("self.%s = %s('%s')"%(n,t,v))

	def _ParseParams(self, param_date):
		# extract the parameters from the db
		nams, vals, typs = QD.GetParams(self.model_id, param_date)
		# since GetParameters extracts everything from the database as a pair of
		# arrays of strings, here we parse them, using the 
		for n,v,t in zip(nams, vals, typs):
			if v == 'None':
				exec("self."+n+" = None")			# bypass type if None was stored
			elif t == 'list':
				exec("self."+n+" = eval(v)")		# list, so use eval to rebuild it
			else:
				exec("self.%s = %s('%s')"%(n,t,v))
	
	def GetParams(self,param_date):
		self._ParseParams(param_date)
		return {'cumdays':self.cumdays, 'tradedir':self.tradedir,'walk_randstate':self.walk_randstate}
		
	def Train(self,train_dates,trade_dates,rndstate=None):
		"""
		other modifications to try:
		---
		add trendstrength filter also?
		---
		measure avg hypothetical gain from being correct on every transition, and instead
		use that (filtered to some threshold) to select how to trade?
		---
		if this is done, with/without attenuation 
		---
		could also use a universe of source and somehow pick subset predictor cum days
		"""
		# start by getting the training data
		d, h, prices = QD.GetTickerPrices_L(self.source_data, train_dates)
		# maybe there are some holes here, so fill (really really shouldn't happen, since I assume target = source)
		prices, filld = QB.FillHoles(prices)
		# get the cumulative counts, and shift it forward to get the next day
		cums = QB.SignCumCount(prices)
		nextday = np.roll(cums,-1,axis=0)
		# drop 1st row (since we don't want to count the shift from 0) and
		# drop the last row (since we dont want to count the circular comparison)
		cums  = cums[1:-1]
		nextday = nextday[1:-1]
		# now get the unique day counts
		unis = np.unique(cums)
		
		# now we can tabulate the transition frequencies: cumcount, freq, persist, reverse
		transfreqs = np.zeros((len(unis),4),dtype=float)
		transfreqs[:,0] = unis
		transfreqs[:,2] = unis + np.sign(unis)
		transfreqs[:,3] = -1*np.sign(unis)
		# must loop through day counts
		for tmp in range(len(unis)):
			transfreqs[tmp,1] = np.sum(cums == transfreqs[tmp,0])
			transfreqs[tmp,2] = np.sum(nextday[cums == transfreqs[tmp,0]] == transfreqs[tmp,2])
			transfreqs[tmp,3] = np.sum(nextday[cums == transfreqs[tmp,0]] == transfreqs[tmp,3])
		# get the ratio of the strong/weak tendencies
		rats = np.max(transfreqs[:,2:],axis=1)/np.min(transfreqs[:,2:],axis=1)
		rats[~(np.isfinite(rats))] = -np.inf		
		
		# figure out which strong/weak disparities to use for signal generation
		disps_to_use = (rats > self.prratio_thresh) & (transfreqs[:,1] > self.cumdays_thresh*len(prices))
		# build the trading rules: if the cumulative directional day is this...
		cumdays = transfreqs[disps_to_use,0]
		# ... then take a postion in this direction
		tradedir = np.sign((transfreqs[disps_to_use,2]-transfreqs[disps_to_use,3])*transfreqs[disps_to_use,0])

		print(QB.PrintTable(np.hstack((transfreqs,np.reshape(rats,(len(unis),1)),np.reshape(disps_to_use,(len(unis),1)))),\
			['%d','%d','%d','%d','%0.4f','%d'],['Count','Freq','Persist','Reverse','Ratios','Chosen']))

		# model is finished, so store the paramters
		param_names = ['cumdays','tradedir','walk_randstate']
		param_values = [list(cumdays.astype(int)),list(tradedir.astype(int)),rndstate]
		param_types = ['list','list','int']
		# now store	
		self._SaveParams(trade_dates[self.buffer_days:], param_names, param_values, param_types)
		return True

	def Trade(self,buffertradedates):
		# get parameters to use
		self._ParseParams(buffertradedates[-1])
		
		# first, check if cumdays is []
		if len(self.cumdays) == 0:
			self.last_signal = 0
		else:
			# start by getting the training data
			d, h, prices = QD.GetTickerPrices_L(self.target_index, buffertradedates)
			# maybe there are some holes here, so fill
			prices, filld = QB.FillHoles(prices)
			# get the cumulative counts
			cums = QB.SignCumCount(prices)
			# find cum in-direction day of the day before the trade day, in the list of those being used
			# and index that into the direction (either persist or reverse)
			try:
				self.last_signal = self.tradedir[self.cumdays.index(cums[-2])]
			except ValueError:
				self.last_signal = 0
				sys.exc_clear()
			
		# store the final signal for today, which will be in .last_signal
		self._SaveSignal(buffertradedates[-1])
		return self.last_signal
