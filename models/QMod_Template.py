#QMod_Template
"""
QREDIS Base Model Class - DO NOT USE
JAH 20120923
JAH 20140118 everything has been tested and seems to be working fine with python 3
"""

class QMod_Template:
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
	Models are to be loaded into QREDIS using exec(open(full_path+file_name).read())
	"""

########################## MODEL MAKER SHOULD NOT NEED TO EDIT ANY OF THESE! ###############################
	def __init__(self, model_id=None, target=None, train=None, trade=None, buff=None, source_data=None, settings=None):
		"""
		There are three ways to initialize a model:
		1) New: Pass None for model_id + all required model definition variables + 
			settings dictionary to create a new model.
		2) Existing: Pass only an existing model_id to setup an existing model from
			the QREDIS database.
		3) Copy: Pass an existing model_id + a settings dictionary + whatever other model
			definition variables you want different from the existing model to create
			a (possibly) modified copy.
		"""
		if model_id is None:
			# if model id is None, create a new model with these params - all are required
			if (target is None) or (train is None) or (trade is None) or (source_data is None) or (buff is None) or (settings is None):
				raise TypeError("To create a new model, all parameters and settings are required!")

			self.target_index = target		# ex: 'SP500'
			self.training_days = train		# ex: 20
			self.trading_days = trade		# ex: 5
			self.buffer_days = buff			# ex: 2
			self.source_data = source_data	# ex: "['SP500','NDX','RUT','MID','DJIA']" or maybe something like "in ('MAJOR','VOLATILITY')" to be used with QD.GetTickers(filt_type="...")[0]
			self.descrip = None
			# create the model in the database and get the ID
			self.model_id = QD.CreateModel(self.__class__.__name__, target, train, trade,buff, source_data)
			self.name = '%s%d'%(self.__class__.__name__,self.model_id)
			# save the model settings JAH 20121120 changed to accept input of other settings
			self._SetMe(settings)
			print("ID = %d; Don't forget to store a description using .Describe('description')!"%self.model_id)
		elif (model_id is not None) and (settings is not None):
			# model id + other settings tuple is provided, so we copy
			# first we get existing model stuff and check it's the same class
			dsc,trg,trn, trd, buf, dat, tmp = QD.GetModel(model_id)
			if tmp != self.__class__.__name__:
				raise ValueError("Model %d name (%s) doesn't match %s!"%(model_id,tmp,self.__class__.__name__))
			# now set my params from either existing model or input
			self.target_index = (target, trg)[target is None]
			self.training_days = (train, trn)[train is None]
			self.trading_days = (trade, trd)[trade is None]
			self.buffer_days = (buff, buf)[buff is None]
			self.source_data = (source_data, dat)[source_data is None]
			# create the model in the database and get the ID
			self.model_id = QD.CreateModel(self.__class__.__name__, self.target_index, self.training_days, \
				self.trading_days, self.buffer_days, self.source_data)
			self.name = '%s%d'%(self.__class__.__name__,self.model_id)
			# copy description
			self.Describe('(COPY) '+dsc)
			# settings - this is REQUIRED
			self._SetMe(settings)
			print("ID = %d; description copied from (%d) - change using .Describe('description')!"%(self.model_id, model_id))
		elif (model_id is not None) and ((target is not None) or (train is not None) or (trade is not None) or\
			(source_data is not None) or (buff is not None)):
			# invalid input: model id + parms but no other, warn user
			raise TypeError("Existing model ID parameter(s) given, but no settings; to copy an existing model, at least the model ID and settings tuple is required!")
		else:
			# only model id given, so get info from the db
			self.model_id = model_id
			self.name = '%s%d'%(self.__class__.__name__,self.model_id)
			self.descrip,self.target_index, self.training_days, self.trading_days, self.buffer_days, self.source_data, tmp\
				= QD.GetModel(model_id)
			# check the name to make sure this model id matches this model class JAH 20121110
			if tmp != self.__class__.__name__:
				raise ValueError("Model %d name (%s) doesn't match %s!"%(model_id,tmp,self.__class__.__name__))
			self._ParseSettings()
		# after everything, ensure there is a folder in QREDIS_out
		global QREDIS_out
		try:
			if not(os.path.exists(QREDIS_out+'/'+self.__class__.__name__)):
				os.mkdir(QREDIS_out+'/'+self.__class__.__name__)
		except Exception as e:
			raise e('Unable to access / create %s subdirectory!'%(QREDIS_out+'/'+self.__class__.__name__))
	
	def __str__(self):
		# JAH added call to _PrintSettings() 20121103, .name on 20121106
		return "%s:\n\tTradeable: %s\n\tUsing data from: %s\n\tTraining Days: %d\n\tTrading Days: %d\n\tBuffer Days: %d\n%s\n%s\nModel Description: %s"\
			%(self.name,self.target_index,self.source_data,self.training_days,self.trading_days,self.buffer_days,'-'*20,self._PrintSettings(),self.descrip)
		
	def _BT(self, signal_date):
		# figure out if we are in backtesting; if QREDIS_Model has not been loaded (NameError)
		# which sets QM.backtest = False, we can't be backtesting... JAH 20121001
		# note that the except condition is NECESSARY but not SUFFICIENT
		# first check if we are backtesting with the Walkforward function in QM
		try:
			if ins.ismodule(QM):
				wfbt = QM.backtest
		except:
			try:
				if ins.ismodule(Qredis_Model):
					wfbt = Qredis_Model.backtest
			except:
				wfbt = False
		if wfbt:
			return 'WF'		# walkforward backtesting
		# maybe can be manual backtesting, so check the date JAH 20121126
		# BT = backtesting, RT = supposed realtime
		return ('RT','BT')[dat.date.today() > signal_date]

	def Describe(self,descrip):
		"""
		Store a useful model description in the QREDIS database. When the model is printed,
		this description is shown.
		"""
		self.descrip = descrip
		return QD.DescribeModel(self.model_id,descrip)

	def _SaveSignal(self,sig_date):
		# run after trade: store signal for this model on this date in the database
		putres = QD.PutSignal(self.model_id, sig_date, self.last_signal, self._BT(sig_date))
		# if not in backtesting run by QM.WalkForward, check if we need to retrain
		if self._BT(sig_date) != "WF":
			self.RetrainCheck()
		return putres

	def GetSignals(self,from_date=None,to_date=None):
		"""
		Extract stored signals from the QREDIS Database. If from_date and to_date are
		not defined, this takes all signals. Otherwise, both dates must be given. This
		function returns a date array and signals array. See QD.GetSignals.
		"""
		if (from_date is None) and (to_date is None):
			return QD.GetSignals(self.model_id)
		else:
			return QD.GetSignals(self.model_id, from_date, to_date)
		return False
		
	def ClearSignals(self,from_date=None,to_date=None):
		"""
		Delete stored signals from the QREDIS Database. If from_date and to_date are
		not defined, this clears all signals. Otherwise, both dates must be given.
		"""	
		if input("!?!ARE YOU SURE YOU WANT TO DELETE MODEL SIGNALS FROM THE DATABASE (YES to proceed)!?!") == "YES":
			if (from_date is None) and (to_date is None):
				return QD.DelSignals(self.model_id)
			else:
				return QD.DelSignals(self.model_id,from_date,to_date)
		else:
			return False

	def _SaveParams(self,param_dates, nams, vals, typs):
		# run after train: store current parameters of this model in the database
		# parameters will be stored for next trading_days dates
		return QD.PutParams(self.model_id, param_dates, nams, vals, typs)
		
	def ClearParams(self,from_date=None,to_date=None):
		"""
		Delete stored model parameters from the QREDIS Database. If from_date and
		to_date are not defined, this clears all parameters. Otherwise, both dates
		must be given.
		"""	
		if input("!?!ARE YOU SURE YOU WANT TO DELETE TRAINING PARAMETERS FROM THE DATABASE (YES to proceed)!?!") == "YES":
			if (from_date is None) and (to_date is None):
				return QD.DelParams(self.model_id)
			else:
				return QD.DelParams(self.model_id,from_date,to_date)
		else:
			return False
	
	def ClearPS(self, from_date=None, to_date=None):
		"""
		Delete stored model parameters and signals from the QREDIS Database. If
		from_date and to_date are not defined, this clears all. Otherwise, both
		dates must be given.
		"""	
		if input("!?!ARE YOU SURE YOU WANT TO DELETE TRAINING PARAMETERS AND SIGNALS FROM THE DATABASE (YES to proceed)!?!") == "YES":
			if (from_date is None) and (to_date is None):
				p = QD.DelParams(self.model_id)
				s = QD.DelSignals(self.model_id)
			else:
				p = QD.DelParams(self.model_id,from_date,to_date)
				s = QD.DelSignals(self.model_id,from_date,to_date)
			return p,s
		else:
			return False, False

	def _SaveSettings(self, nams, vals, typs):
		# save settings, names, and their values to the database JAH 20121103
		return QD.PutSettings(self.model_id, nams, vals, typs)

	def GetSettings(self):
		"""
		Extract stored model settings from the QREDIS Database. This function returns
		three string arrays. see QD.GetSettings.
		JAH 20121103
		"""
		return QD.GetSettings(self.model_id)
		
	def NextTradeBlock(self):
		"""
		Calculate the next trade block for this model by looking in the signals
		table and building the block extrapolated from the next trade day
		JAH 20121122
		"""
		# first get the last trade date
		last_trade = self.GetSignals()[0][-1]
		# get the buffer days - go back buffer days + 30 just to be safe
		tmp = last_trade + dat.timedelta(days=-30 - self.buffer_days)
		d,h = QD.GetTickerCalendar(self.target_index,tmp,last_trade,True)
		# now just extract the last buffer_days dates (excluding holidays of course)
		trade_block = d[~h][-self.buffer_days:]
		# now need to get the next trade date, go ahead 30 days just to be safe
		tmp = last_trade + dat.timedelta(days=30)
		d,h = QD.GetTickerCalendar(self.target_index,last_trade,tmp,True)

		return np.append(trade_block,d[~h][1])
	
	def NextTrainTradeBlock(self):
		"""
		Calculate the next training and trading blocks for this model by looking
		in the parameters table and building the blocks extrapolated from the last
		existing parameter date
		JAH 20121122
		"""
		# first get the last param date; this will be the next last train date
		last_param = QD.GetLastParamsDate(self.model_id)
		# get the training+buffer days - go back extra 30 just to be safe
		tmp = last_param + dat.timedelta(days=-30 - self.buffer_days - self.training_days)
		d,h = QD.GetTickerCalendar(self.target_index,tmp,last_param,False)
		# now just return the last buffer+training dates (excluding holidays of course)
		train_block = d[~h][-(self.buffer_days+self.training_days):]
		# now get the trade block
		tmp = last_param + dat.timedelta(days = self.trading_days+30)
		d,h = QD.GetTickerCalendar(self.target_index,last_param+dat.timedelta(days=1),tmp,False)
		# still need to add on the buffer days, but will just take that from training block
		
		return train_block, np.append(train_block[-self.buffer_days:],d[~h][:self.trading_days])

	def RetrainCheck(self):
		"""
		Check if the last signal is on (or after, though should never happen) the
		last parameter date, if so, it means the model should be retrained
		JAH 20121122
		"""
		sd = QD.GetSignals(self.model_id)[0][-1]
		pd = QD.GetLastParamsDate(self.model_id)
		if sd > pd:
			print('!!!Something wrong, last signal date %s is AFTER last parameter date %s!!!'%(sd,pd))
		elif sd == pd:
			print('Time to retrain, last signal date %s is SAME as last parameter date %s!'%(sd,pd))
		
		return sd >= pd
		
########################## MODEL MAKER SHOULD EDIT THESE! ###############################

	def _SetMe(self, mysettings):
		# THIS SHOULD ONLY BE RUN AS PART OF THE FIRST MODEL INITIALIZATION
		# DURING MODELING, USE _ParseSettings TO SET THEM FROM THE DB!!!
		# define and save model settings (settings as opposed to parameters, which can
		# change over time; settings are constant
		return True
	
	def _PrintSettings(self):
		# this prints the model settings; it is called from the __str__ function
		return ''
		
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
		for n,t,v in zip(nams, typs, vals):
			exec("self.%s = %s('%s')"%(n,t,v))
	
	def GetParams(self,param_date):
		"""
		Extract stored model parameters from the QREDIS Database for a specified
		date, which is required. Returns a dictionary holding the parameters.
		JAH 20121128
		"""
		self._ParseParams(param_date)
		return None

	def GetSource(self):
		"""
		The source_data has been stored as just a string; parse it into (probably)
		a list or array of tickers.
		JAH 20130526
		"""

	def Train(self,train_dates,trade_dates,rndstate=None):
		"""
		Look at some data over a specified training block, then generate some kind
		of modeling parameters to store in the QREDIS database.  These are then used
		by the Trade method on subsequent day(s) to generate signals.
		"""
		return True

	def Trade(self,buffertradedates):
		"""
		Read some model parameters generated by a previous execution of the Train
		method, then generate some model to execute trading signals that are then
		stored in the QREDIS database.
		"""	
		return 1
