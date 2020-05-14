#QMod_PersReverRet_Select
"""
QREDIS Persistence Reverse Return Select model
"""

class QMod_PersReverRet_Select(QMod_Template):
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
	of these tendencies to trade during the trading period.  The # of occurences threshold
	is a % of the training period, and the tendency is measured as the direction of the net
	effect of holding a position for each in-direction day being larger than a specified
	value. If attenuate=True, the daily returns for this net effect calculation are
	attenuated.  Remember that attenuation will *dramatically* reduce the cumulative
	moves over a long period of time, so neteffect should be lower than you would think,
	if you decide to use attenuation.
	The settings dict should have elements:
		'neteffect_thresh' (float), example = 0.01
		'cumdays_thresh' (float), example = 0.1
		'atten' (bool), example = True
	Load using exec(open(QREDIS_mod+'/QMod_PersReverRet_Select.py').read())
	"""

########################## MODEL MAKER SHOULD EDIT THESE! ###############################

	def _SetMe(self, mysettings):
		# THIS SHOULD ONLY BE RUN AS PART OF THE FIRST MODEL INITIALIZATION
		# DURING MODELING, USE _ParseSettings TO SET THEM FROM THE DB!!!
		# define and save model settings (settings as opposed to parameters, which can
		# change over time; settings are constant
		self.neteffect_thresh = mysettings['neteffect_thresh']	# threshold for a net effect to be "significant"
		self.cumdays_thresh = mysettings['cumdays_thresh']		# necessary min observations (%) of a cum day count
		self.atten = mysettings['atten']						# attenuate daily returns
		# now save		
		self._SaveSettings(['neteffect_thresh','cumdays_thresh','atten'],\
		[self.neteffect_thresh,self.cumdays_thresh,self.atten], ['float','float','bool'])
		return True
	
	def _PrintSettings(self):
		# this prints the model settings; it is called from the __str__ function
		return '\tThresholds=\n\tNet Gain: %0.4f\n\tCum Day Perc: %0.2f\n\tAttenuate: %s'%(self.neteffect_thresh, self.cumdays_thresh,self.atten)
		
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

		# compute the daily returns and roll forward so they line up with nextday
		dailyrets = roll((QB.Diff(prices,1)/QB.LagLead(prices,1))[1:-1],-1)
		# and possibly attenuate
		if self.atten:
			dailyrets = QB.Attenuate(dailyrets)
		
		# now we can tabulate the transition frequencies: cumcount, freq, persist cnt & ret, reverse  cnt & ret, net
		transfreqs = np.zeros((len(unis),7),dtype=float)
		transfreqs[:,0] = unis
		transfreqs[:,2] = unis+np.sign(unis)
		transfreqs[:,4] = -1*np.sign(unis)
		# must loop through day counts
		for tmp in range(len(unis)):
			transfreqs[tmp,1] = np.sum(cums == transfreqs[tmp,0])
			# get the persists
			d = (cums == transfreqs[tmp,0]) & (nextday == transfreqs[tmp,2])
			transfreqs[tmp,3] = np.abs(np.sum(dailyrets[d]))
			transfreqs[tmp,2] = np.sum(d)
			# get the reverses
			d = (cums == transfreqs[tmp,0]) & (nextday == transfreqs[tmp,4])
			transfreqs[tmp,5] = np.abs(np.sum(dailyrets[d]))
			transfreqs[tmp,4] = np.sum(d)
			# finally the net effect - if the moves tended to be up, the sum will probably be +, suggesting to go long
			transfreqs[tmp,6] = np.sum(dailyrets[cums == transfreqs[tmp,0]])
		
		# finally, filter based enough days experience and net effect > threshold
		disps_to_use = (np.abs(transfreqs[:,6]) > self.neteffect_thresh) & (transfreqs[:,1] > self.cumdays_thresh*len(prices))

		# build the trading rules: if the cumulative directional day is this...
		cumdays = transfreqs[disps_to_use,0]
		# ... then take a postion in this direction
		tradedir = np.sign(transfreqs[disps_to_use,6])

		print(QB.PrintTable(np.hstack((transfreqs,np.reshape(disps_to_use,(len(unis),1)))),\
			['%d','%d','%d','%0.2f','%d','%0.2f','%0.2f','%d'],\
			['Count','Freq','Pers. #','Pers. $','Reve. #','Reve. $','Net','Trade']))
	
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
