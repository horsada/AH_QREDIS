#QREDIS_Model
"""
QREDIS model execution and evaluation functions
----------------------
ModelEval - 20121103 - for a model and date range, generate the trade history, poop sheet, and trade plot
TradePOOP - 20121006 - generate POOP statistics and optionally print a POOP sheet for a specified model
ThreshTrade - 20121104 - convert float array to signals and simulate trading
TradeHistory - 20121002 - generate a history of daily trade returns for a specified model
TradePlot - 20121011 - plot compounded daily returns curves for a specified model
WalkForward - 20121001 - walk-forward backtest a specified model
JAH 20140118 everything has been tested and seems to be working fine with python 3
----------------------
"""

import sys
import math
import string
import datetime as dat
import numpy as np
import scipy.stats as sstat
import matplotlib.pyplot as plt
import QREDIS_Basic as QB
import QREDIS_Data as QD

backtest = False

def WalkForward(this_model, from_date, to_date, print_trades=True, rand_state=None, saveoutput=False):
	"""
	Backtest walking forward a model in a specified date range. The dates used are controlled
	by the calendar dates for the target index.  Signals will only be generated and stored
	for these dates.  The parameter rand_state is useful testing different configurations and
	settings for models that make use of stochastic algorithms such as the GA. The used date
	range may be different than that specified because of holidays, missing data at endpoints,
	or the requirement to have an integer number of complete training/trading periods.
	---
	Usage: date_range = WalkForward(this_model, from_date, to_date, print_trades, rand_state, saveoutput)
	---
	this_model: this will be assumed to be an initialized model having at least attributes
		target_index, training_days, trading_days, buffer_days, and methods Train() and Trade()
	from_date / to_date: datetime.date first / last day required (inclusive)
	print_trades: True* = print information on each trade, False = don't
	rand_state*: simply gets passed to the model's Train method
	saveoutput: True: save all files to QREDIS_mod / model-specific folder; False*: don't
	date_range: tuple holding the first + last training days and the first + last trading days
	---
	ex: 
	JAH 20121001
	"""
	
	# ensure input variables are correct; out_path added JAH 20121031
	if (type(from_date) is not dat.date) or (type(to_date) is not dat.date):
		raise TypeError("Dates must be datetime.date: %s"%WalkForward.__doc__)
	if type(saveoutput) is not bool:
		raise TypeError('Variable saveoutput must be True/False: %s'%WalkForward.__doc__)
	
	# start timing & talking; printing to diary added JAH 20121031
	stt = dat.datetime.now()
	if saveoutput:
		# JAH 20140118 changed to calculate out_path		
		out_path = __builtins__['QREDIS_out']+'/'+this_model.name[:this_model.name.index(str(this_model.model_id))]+'/'
		save_file = '%sQMWalk_%s_%s.txt'%(out_path,this_model.name,QB.TimeStamp(stt))
		QB.Diary(save_file)
	print("WalkForward begun for %s^ from %s to %s on %s\n%s"%(this_model.name, from_date, to_date,stt,"-"*50))
	print('^%s\n%s'%(this_model	,"-"*50))
	global backtest; backtest = True
	
	# first get the last non-0 price date for the target
	lst_dat, lst_cls = QD.GetTickerLastClose(this_model.target_index)
	lst_dat = min(to_date, lst_dat)
	
	# get the trading days
	alldats, target_holis = QD.GetTickerCalendar(this_model.target_index, from_date, to_date, trim=True)
	target_dates = alldats[~target_holis]
	n = len(target_dates)
	trade_days = n - this_model.training_days - this_model.buffer_days
	
	# determine how many days we need and if we must extend the dates given JAH 20121122
	# get the integer number of training/trading periods we can do with this calendar
	trdtrn_steps = math.ceil(trade_days/float(this_model.trading_days))
	# if not enough days in calendar, the final trade_block won't be large enough to store
	# in the database all the params for the next trading_days days
	days_missing = abs(trade_days - this_model.trading_days*trdtrn_steps)
	# adjust the walkforward calendar if we need to add extra days
	if days_missing > 0:
		# have to take extra in case holidays come after the end
		alldats, target_holis = QD.GetTickerCalendar(this_model.target_index, target_dates[0], target_dates[-1]+dat.timedelta(days=30), trim=False)
		# now take holidays out and get JUST the days I need
		target_dates = alldats[~target_holis][:(n+days_missing)]
		trade_days = len(target_dates) - this_model.training_days - this_model.buffer_days
		print('!!! Walkforward period extended by %d days !!!'%days_missing)
	
	# now loop and do the training & trading
	this_day = 0
	while this_day <= trade_days:
		# get the training block JAH added calc for buffer days 20121102
		train_block = target_dates[this_day:(this_day+this_model.training_days+this_model.buffer_days)]
		
		# now get the trading block JAH added calc for buffer days 20121102
		trade_block = target_dates[(this_day+this_model.training_days):\
			(this_day+this_model.training_days+this_model.trading_days+this_model.buffer_days)]
		# make sure we have enough trading days
		if len(trade_block) <= this_model.buffer_days:
			break

		# now train
		print("Training model %s from %s to %s"%(this_model.name,train_block[0],train_block[-1]))
		this_model.Train(train_block,trade_block, rand_state)

		# trade & increment day counter
		for cnt in range(this_model.buffer_days,len(trade_block)):
			# increment trade counter
			this_day += 1
			# get the dates from buffer days through today
			td = trade_block[cnt-this_model.buffer_days:(cnt+1)]
			# trade!!!
			this_model.Trade(td)
			if print_trades:
				print("\tTrading model %s on %s(%s): result = %d"%(this_model.name,td[-1],td[0],this_model.last_signal))
			# see if we just traded our last day
			if td[-1] >= lst_dat:
				this_day = trade_days
				break

	backtest = False
	
	print('Model Trained from %s to %s!\nModel Traded from %s to %s!'%\
		(target_dates[0],target_dates[-1],target_dates[this_model.training_days+this_model.buffer_days],min(td[-1],lst_dat)))
	
	# finish up
	stp = dat.datetime.now()
	print("WalkForward ended for model %s on %s\nExecution time: %s\n%s"%(this_model.name,stp,stp-stt,"-"*50))
	if saveoutput:
		print('WalkForward output save in %s'%save_file)
		QB.Diary(None)
	
	return target_dates[0],target_dates[-1],target_dates[this_model.training_days+this_model.buffer_days],min(td[-1],lst_dat)

def TradeHistory(this_model, from_date=None, to_date=None, StressParams = None):
	"""
	Take the signals from a model over a specified time horizon and create a set of
	trading histories. This also generates the accumulated trade history of the
	target index for comparison. Note that this assumes there are signals in the entire
	date range	up to (inclusive) the to date! The model can be stressed in 3 ways.
	The first is to remove the top x% of biggest up days in the target by replacing their
	moves with the overall average daily move. Similarly, it can be stressed by smoothing
	over the top y% of biggest down days. Finally, stress the model by reversing or
	nullifying (if 1-direction model) the top z% of best trades.
	---
	Usage: dates, signals, target returns, total, long, short, target, stressed = TradeHistory(this_model, from_date, to_date, StressParams)
	---
	this_model: this will be assumed to be an initialized model having at least the target_index attribute
	from_date* / to_date*: datetime.date first / last signal day; if None is 
		passed for either, the entire signal history is used
	StressParams: array_like holding 3 values: % of Top Up Days to smooth,
		% of Top Down Days to smooth, % of best trades to reverse or nullify
	dates: actual dates (from date + 1 : to date inclusive) for trades
	signals: array of model signals used (stressed, possibly)
	target returns: aray of daily returns for the target (stressed, possibly)
	long: array of accumulated daily trade returns for the long signals only
	short: array of accumulated daily trade returns for the short signals only
	total: array of accumulated daily trade returns for the total model
	target: array of accumulated daily buy-and-hold returns for target index
	stressed: nested list holding for each type of stress the dates to which the
		stress was actually applied
	---
	ex: 
	JAH 20121002
	"""
	
	# if dates not input, just get the first and last signal date from the database JAH 20121106
	if (from_date is None) or (to_date is None):
		signal_dates, signal_values = this_model.GetSignals()
		from_date = signal_dates[0]
		to_date = signal_dates[-1]
	
	# ensure input variables are correct
	if (type(from_date) is not dat.date) or (type(to_date) is not dat.date):
		raise TypeError("Dates must be datetime.date: %s"%TradeHistory.__doc__)
		
	# check stress parameters JAH 20121011
	Stressd = [None]*3		# will store dates stressed in each type (Top, Bottom, Signal)
	try:
		(TopRemovePerc, BotRemovePerc, SigRemovePerc) = StressParams
	except TypeError:
		if StressParams is not None:
			print("Something wrong with StressParams, just using 0s...")
		TopRemovePerc = 0.0
		BotRemovePerc = 0.0
		SigRemovePerc = 0.0
	
	# check parameters for correctness
	if (type(TopRemovePerc) is not float) or (TopRemovePerc >= 1) or (TopRemovePerc < 0) or\
		(type(BotRemovePerc) is not float) or (BotRemovePerc >= 1) or (BotRemovePerc < 0) or\
		(type(SigRemovePerc) is not float) or (SigRemovePerc >= 1) or (SigRemovePerc < 0):
		raise ValueError("Stress parameter percentages must be floats in (0,1) : %s"%TradeHistory.__doc__)

	# get the signals first, but only if we didn't already to it; JAH 20121106
	try:
		signal_dates
	except NameError:
		# if it get here we did NOT already extract all signals to get the from/to dates
		signal_dates, signal_values = this_model.GetSignals(from_date, to_date)
	
	# check the target calendar, so we can get the first signal_day - 1
	# we take from_date - 30 to hopefully ensure that after holidays + weekends,
	# from_date-1 will be in date[not holidays][-2] JAH 20121106
	a, t = QD.GetTickerCalendar(this_model.target_index, from_date+dat.timedelta(days=-30), from_date)
	
	# next get the target index and remove the holidays
	target_dates,target_holis,target_prices = QD.GetTickerPrices_R(this_model.target_index, a[~t][-2], to_date)
	dats = target_dates[~target_holis].flatten()

	# check that the dates seem like they line up on the boundaries at least
	# 1st signal date should be 2nd trade (2nd price) date AND last dates should match AND price dates 1 longer
	if (signal_dates[0] != dats[1]) or (signal_dates[-1] != dats[-1]) or (len(signal_dates)+1 != len(dats)):
		print("Signal Dates: %s, "%signal_dates)
		print("Price Dates: %s, "%dats)
		print("Trade Dates: %s, "%dats[1:])
		raise ValueError("Signal and price dates don't seem to line up!")	
	dats = dats[1:]
	
	# get rid of holidays, and fill in any holes
	target_prices,filld = QB.FillHoles(target_prices[~target_holis])
	# maybe there are holes on the first day wich will cause an annoying error
	# in the divsion, so fill in nans, I guess
	target_prices[0,target_prices[0,:] == 0] = np.nan
	# finally, flatten
	target_prices = target_prices.flatten()
	
	# compute returns - this automatically drops the first item
	returns = np.diff(target_prices,1,axis=0)/(target_prices[:-1])
	# by now the signals daterange and prices daterange[1:] should line up
	
	if StressParams is not None:	# JAH 20121011
		# STRESS the target index: replace the biggest moves with the average but in the same direction
		n = len(dats)
		avgret = np.mean(returns)
		# get the biggest move indices (remember sorts ascending)
		bot_to_top = np.argsort(returns)
		# replace the biggest down moves...
		rem = bot_to_top[:int(BotRemovePerc*n)]
		returns[rem] = -avgret
		Stressd[0] = dats[rem]	# save the dates stressed
		# ...  then replace the biggest up moves
		rem = bot_to_top[(n-int(TopRemovePerc*n)):]
		returns[rem] = avgret
		Stressd[1] = dats[rem]	# save the dates stressed
		
		# STRESS the signals: reverse (or nullify) the best signals
		tot = np.cumsum(signal_values*returns)	# get uncompounded daily gains
		# get the signals to reverse or nullify
		rem = np.argsort(tot)[(n-int(SigRemovePerc*np.sum(signal_values!=0))):]
		# maybe this is a uni-directional model, in which case we need to nullify and not reverse
		if (np.sum(signal_values == 1) == 0) or (np.sum(signal_values == -1) == 0):
			signal_values[rem] = 0
		else:
			signal_values[rem] = -1*signal_values[rem]
		Stressd[2] = dats[rem]	# save the dates stressed
	
	# finally we can trade!
	tot = np.cumproduct(1+signal_values*returns)
	lng = signal_values.copy(); lng[lng == -1] = 0; lng = np.cumproduct(1+lng*returns)
	sht = signal_values.copy(); sht[sht == 1] = 0; sht = np.cumproduct(1+sht*returns)
	tar = np.cumproduct(1+returns)
		
	return dats,signal_values,returns,tot,lng,sht,tar,Stressd

def TradePOOP(this_model, dats, signals, ret_target, StressParams = None, POOPPrint = True, saveoutput = False):
	"""
	Take the signals from a model over a specified time horizon and create a set of
	After running TradeHistory, pass the dates, signals, and daily target index returns
	and this will compute the POOP on the model and if desired print a nicely-formatted
	POOP sheet to the console. The model stats are returned in a dictionary, and nice
	printable names of the stats are in another dictionary, both using the same keys.
	Data on the top drawdowns are returned in a separate nested list
	---
	Usage: model stats, stat names, drawdowns, save file= TradePOOP(this_model, dats, signals, ret_target, StressParams, POOPprint, saveoutput)
	---
	this_model: this will be assumed to be an initialized model having at least the target_index attribute
	dats: array of actual trade dates SHOULD COME FROM TradeHistory
	signals: array of trade signals SHOULD COME FROM TradeHistory
	ret_target: array of daily target returns SHOULD COME FROM TradeHistory
	StressParams: stress parameters used for TradeHistory
	POOPprint: True* = print POOP sheet, False = don't
	saveoutput: save POOP sheet to the  QREDIS_mod / model-specific folder
		if this ans POOPprint are True; False*: don't
	model stats: dictionary of model statistics
	stat names: dictionary model stat names with same keys as model stats
	drawdown: nested list of stats for top drawdowns: start date, maximum loss from peak, 
		days to bottom, days to recover, last date, current flag
	save file: full path & filename of output POOP sheet (if any)
	---
	ex: 
	JAH 20121006
	"""
	
	# JAH 20121011 changed to take inputs from TradeHistory instead of doing all the calcs again
	# in theory, this is taking the dats, signals, and ret_target arrays from TradeHistory, but do *some* checking
	# JAH 20121031 added test for out_path
	lls = [len(dats),len(signals),len(ret_target)]
	if (type(dats) is not np.ndarray) or (type(signals) is not np.ndarray) or (type(ret_target) is not np.ndarray):
		raise TypeError("Dates, Signals, and Target should all be arrays: %s"%TradePOOP.__doc__)
	if not(all([x == lls[0] for x in lls])):
		raise ValueError("Dates, Signals, and Target should all be same length: %s"%TradePOOP.__doc__)
	if type(saveoutput) is not bool:
		raise TypeError('Variable saveoutput must be True/False: %s'%TradePOOP.__doc__)
	
	# check stress parameters by trying to unpack JAH 20121011 new, also stress params printed with POOP at bottom is new
	try:
		(TopRemovePerc, BotRemovePerc, SigRemovePerc) = StressParams
	except TypeError:
		if StressParams is not None:
			print("Something wrong with StressParams, just using 0s...")
		TopRemovePerc = 0.0
		BotRemovePerc = 0.0
		SigRemovePerc = 0.0
	
	# split the target daily returns into ups & downs
	ret_target_up = np.abs(ret_target*(ret_target>0))	# target up day returns (abs is because this can have -0.'s)
	ret_target_dn = ret_target*(ret_target<0)				# target down day returns
	
	# create the daily returns vectors
	ret_model = signals*ret_target					# daily return history for model
	ret_long = (signals == 1)*ret_target			# daly return history for long model
	ret_short = -1*(signals == -1)*ret_target		# daily return history for short model
		
	# signal statistics
	trade_days = signals.size						# number of days traded
	days_long = np.sum(signals == 1)				# number of days model long
	days_short = np.sum(signals == -1)			# number of days model short
	days_invested = days_long + days_short			# number of days traded
	days_flat = trade_days - days_invested			# number of days out of market
	days_chgd = np.sum(signals != np.roll(signals,1)) - (signals[0]!=signals[-1]) # of days position changed
	
	# model vs. target stats
	jnk = ret_model[ret_model > 0]
	days_model_up = jnk.size						# number of up days for model
	avg_model_up = np.mean(jnk)					# average up day for model
	jnk = ret_model[ret_model < 0]
	days_model_dn = jnk.size						# number of days model was down
	avg_model_dn = np.mean(jnk)					# average down day for model
	jnk = ret_target[ret_target > 0]
	days_target_up = jnk.size						# number of days target was up
	avg_target_up = np.mean(jnk)					# average up day for target
	jnk = ret_target[ret_target < 0]
	days_target_dn = jnk.size						# number of days target was down
	avg_target_dn = np.mean(jnk)					# average down day for target
	uncomp_up_capture = np.sum(ret_target_up*(signals > 0))/np.sum(ret_target_up)				# target up capture
	uncomp_dn_capture = max(0,-np.sum(-ret_target_dn*(signals < 0))/np.sum(ret_target_dn))	# target down capture
	correl =  sstat.pearsonr(ret_target, ret_model)[0]		# correlation between daily returns
	up_correl = sstat.pearsonr(ret_target_up,ret_long)[0]	# correlation for up days
	dn_correl = sstat.pearsonr(ret_target_dn,ret_short)[0]	# correlation for down days
	
	# accumulated returns
	cum_model = np.cumproduct(1+ret_model)
	cum_long = np.cumproduct(1+ret_long)
	cum_short = np.cumproduct(1+ret_short)
	cum_target = np.cumproduct(1+ret_target)	
	
	# return stats
	avg_model = np.mean(ret_model)				# model average daily return
	std_model = np.std(ret_model)				# std dev of daily returns
	std_target = np.std(ret_target)				# std dev of target daily returns
	ann_std_model = std_model*math.sqrt(250)		# std. dev is annualized by multiplying by sqrt(# trading days)
	comp_model = cum_model[-1]-1					# total compound model return
	comp_long = cum_long[-1]-1						# compound long model return
	comp_short = cum_short[-1]-1					# compound short model return
	comp_target = cum_target[-1]-1					# compound target return
	ann_compROR_model = (comp_model+1)**(250.0/trade_days)-1		# annualized compounded rate of return for model
	ann_compROR_target = (comp_target+1)**(250.0/trade_days)-1		# annualized compounded rate of return for target
	riskfree = 0.05													# risk free rate for Sharpe
	sharpe = (ann_compROR_model - riskfree)/ann_std_model			# Sharpe ratio
	
	# WHOO HOO time to put everything together!
	model_stats = {'ann_compROR_model':ann_compROR_model,'ann_compROR_target':ann_compROR_target,\
		'sharpe':sharpe,'comp_model':comp_model,'comp_long':comp_long,'comp_short':comp_short,\
		'comp_target':comp_target,'avg_model':avg_model,'std_model':std_model,'ann_std_model':ann_std_model,\
		'trade_days':trade_days,'days_invested':days_invested,'days_flat':days_flat,'days_long':days_long,\
		'days_short':days_short,'correl':correl,'up_correl':up_correl,'dn_correl':dn_correl,\
		'uncomp_up_capture':uncomp_up_capture,'uncomp_dn_capture':uncomp_dn_capture,'days_model_up':days_model_up,\
		'avg_model_up':avg_model_up,'days_target_up':days_target_up,'avg_target_up':avg_target_up,\
		'days_model_dn':days_model_dn,'avg_model_dn':avg_model_dn,'days_target_dn':days_target_dn,\
		'avg_target_dn':avg_target_dn,'std_target':std_target,'days_chgd':days_chgd}
	stat_names = {'ann_compROR_model':'Annualized Compound Model ROR','ann_compROR_target':'Annualized Compound Target ROR',\
		'sharpe':'Sharpe Ratio','comp_model':'Compounded Model Return','comp_long':'Compounded Long Return','comp_short':'Compounded Model Return',\
		'comp_target':'Compounded Target Return','avg_model':'Daily Return Average','std_model':'Daily Return Standard Deviation','ann_std_model':'Annualized Standard Deviation',\
		'trade_days':'# Days Model Run','days_invested':'# Days Invested','days_flat':'# Days Flat','days_long':'# Days Long',\
		'days_short':'# Days Short','correl':'Correlation','up_correl':'Up Correlation','dn_correl':'Down Correlation',\
		'uncomp_up_capture':'Uncompounded Up Capture','uncomp_dn_capture':'Uncompounded Down Capture','days_model_up':'Model # Days Up',\
		'avg_model_up':'Model Up Average Gain','days_target_up':'Target # Days Up','avg_target_up':'Target Up Average Gain',\
		'days_model_dn':'Model # Days Down','avg_model_dn':'Model Down Average Loss','days_target_dn':'Target # Days Down',\
		'avg_target_dn':'Target Down Average Loss','std_target':'Target Daily Return Standard Deviation',\
		'days_chgd':'# Days Trade Position Changed'}
	
	# start the drawdown calculations
	maxs = np.maximum.accumulate(cum_model)		# cumulative max balance
	maxs[0] = max(1,maxs[0])						# maybe 1st trade can be bad, so force to start a dd here
	drawdowns = maxs-cum_model						# difference from curr balance to max balance
	dd = np.sign(drawdowns)						# in a drawdown signal
	
	# figure out where the drawdowns start and stop
	# start: find where a day is a drawdown but the previous day is not
	ddstarts = np.where((dd == 1) * (np.roll(dd,1) == 0))[0]
	# stop: find where a day is not a drawdown but the previous day is
	# then drop the 1st if it was a drawdown rolled around, and add the last trade day if it is a drawdown
	ddstops =  np.where((dd == 0) * (np.roll(dd,1) == 1))[0]
	# check if the last day is a drawdown, if so we need to fix the stops array
	if dd[-1] == 1:
		ddstops = np.append(ddstops[1:],trade_days)
		
	# now we loop through the all the drawdowns - we want the top DDnum, so save all trough values
	# we will have to save all the stats, then take the top DDnum and save only them finally
	DDnum = 5; mxs = []; ddstats = []; currentDD = False
	for stt,stp in zip(ddstarts,ddstops):
		ddrng = list(range(stt,stp))
		# first get the trough from this drawdown
		mx = max(drawdowns[ddrng])/maxs[stt]
		# then save it
		mxs.append(-mx)
		# start day, max drawdown (relative), days to trough, days to recover, last day, current drawdown
		ddstats.append([dats[stt],mx,np.argmax(drawdowns[ddrng])+1,stp-stt,dats[stp-1],stp==trade_days])
		currentDD = (currentDD or stp==trade_days)
		
	# get the top DDnum drawdowns in order (using -mxs since argsort sorts ascending)
	ddindx = np.argsort(mxs)[:DDnum]
	# now move the desired drawdown stats into the final list
	DDstats = [None]*min(DDnum,len(mxs)	)
	for jnk in range(len(ddindx)):
		DDstats[jnk] = ddstats[ddindx[jnk]]	
	
	# print poop?
	if POOPPrint:
		# first let's get the POOP on the model
		thismodel = '%sPOOP SHEET FOR %s^'%(("STRESSED ","")[StressParams == None],this_model.name)
		moddescrip = '^%s'%this_model
		# rows should be this long the extra space will be _'s
		rowlen = 75; blanks = '_'*rowlen; lins = '-'*rowlen
		# first we build a row to see how long it is, then insert the _'s
		row1 = 'Period Traded:%s%s to %s'%('%s',dats[0],dats[-1])
		row1 = row1%(blanks[:(rowlen-len(row1)+2)])
		row2 = 'Annualized Compound Model ROR:%s%0.2f%%%%'%('%s',ann_compROR_model*100)
		row2 = row2%(blanks[:(rowlen-len(row2)+3)])
		row3 = 'Annualized Compound Index ROR:%s%0.2f%%%%'%('%s',ann_compROR_target*100)
		row3 = row3%(blanks[:(rowlen-len(row3)+3)])
		row4 = 'Sharpe Ratio (Annual Risk Free = %0.2f%%%%):%s%0.3f'%(riskfree*100,'%s',sharpe)
		row4 = row4%(blanks[:(rowlen-len(row4)+3)])
		# drawdowns table header
		rowDDH = '%s%d Worst Drawdowns%s%s'%('%s',min(len(DDstats),DDnum),['','*(currently in drawdown)'][currentDD],'%s')
		jnk = len(rowDDH)
		if (rowlen-jnk+4) % 2 == 0:
			# evenly divisible so just split it JAH 20141118 force int division since in python3 this will be float
			rowDDH = rowDDH%(blanks[:(rowlen-jnk+4)//2],blanks[:(rowlen-jnk+4)//2])
		else:
			# put 1 extra on the front: here I rely on the fact that dividing an odd int by 2 rounds down
			# JAH 20121221, in preparation for future use of 3.x, force integer division with //, since floor
			# division will no longer be the default
			rowDDH = rowDDH%(blanks[:((rowlen-jnk+4)//2+1)],blanks[:(rowlen-jnk+4)//2])
		# build the rows for the drawdowns
		rowDDs = ''
		col15len = 15; colslen = (rowlen - 2*col15len)//3	# JAH 20140118 force integer division
		for jnk in DDstats:
			thisrowa = '%s%s'%(jnk[0],'%s'); thisrowa = thisrowa%(blanks[:(col15len-len(thisrowa)+2)])
			thisrowb = '%0.2f%%%%%s'%(jnk[1]*100,'%s'); thisrowb = thisrowb%(blanks[:(colslen-len(thisrowb)+3)])
			thisrowc = '%d%s'%(jnk[2],'%s'); thisrowc = thisrowc%(blanks[:(colslen-len(thisrowc)+2)])
			thisrowd = '%d%s'%(jnk[3],'%s'); thisrowd = thisrowd%(blanks[:(colslen-len(thisrowd)+2)])
			thisrowe = '%s%s%s'%('%s',jnk[4],['','*'][jnk[5]]); thisrowe = thisrowe%(blanks[:(col15len-len(thisrowe)+2)])
			rowDDs = '\n'.join((rowDDs,thisrowa+thisrowb+thisrowc+thisrowd+thisrowe))
		# add the column descriptions
		thisrowa = 'First Day%s'%('%s'); thisrowa = thisrowa%(blanks[:(col15len-len(thisrowa)+2)])
		thisrowb = 'Max %%%% Lost%s'%('%s'); thisrowb = thisrowb%(blanks[:(colslen-len(thisrowb)+3)])
		thisrowc = 'Days to Trough%s'%('%s'); thisrowc = thisrowc%(blanks[:(colslen-len(thisrowc)+2)])
		thisrowd = 'Days to Recover%s'%('%s'); thisrowd = thisrowd%(blanks[:(colslen-len(thisrowd)+2)])
		thisrowe = '%sLast Day'%('%s'); thisrowe = thisrowe%(blanks[:(col15len-len(thisrowe)+2)])
		rowDDD = thisrowa+thisrowb+thisrowc+thisrowd+thisrowe
		# continuing...
		row5 = 'Compounded Model Return:%s%0.2f%%%%'%('%s',comp_model*100)
		row5 = row5%(blanks[:(rowlen-len(row5)+3)])
		row6 = 'Compounded Long Return:%s%0.2f%%%%'%('%s',comp_long*100)
		row6 = row6%(blanks[:(rowlen-len(row6)+3)])
		row7 = 'Compounded Short Return:%s%0.2f%%%%'%('%s',comp_short*100)
		row7 = row7%(blanks[:(rowlen-len(row7)+3)])
		row8 = 'Compounded Index Return:%s%0.2f%%%%'%('%s',comp_target*100)
		row8 = row8%(blanks[:(rowlen-len(row8)+3)])
		row9 = 'Average Daily Return:%s%0.2f%%%%'%('%s',avg_model*100)
		row9 = row9%(blanks[:(rowlen-len(row9)+3)])
		row10 = 'Daily Return Standard Deviation:%s%0.2f%%%%'%('%s',std_model*100)
		row10 = row10%(blanks[:(rowlen-len(row10)+3)])
		row10a = 'Target Daily Return Standard Deviation:%s%0.2f%%%%'%('%s',std_target*100)
		row10a = row10a%(blanks[:(rowlen-len(row10a)+3)])		
		row11 = 'Annualized Standard Deviation:%s%0.2f%%%%'%('%s',ann_std_model*100)
		row11 = row11%(blanks[:(rowlen-len(row11)+3)])
		row12 = 'Number of Days Model Run:%s%d'%('%s',trade_days)
		row12 = row12%(blanks[:(rowlen-len(row12)+2)])
		row12a = 'Number of Days Trade Position Changed:%s%d'%('%s',days_chgd)
		row12a = row12a%(blanks[:(rowlen-len(row12a)+2)])
		row13 = 'Days Invested:%s%0.2f%%%%'%('%s',100.0*days_invested/trade_days)
		row13 = row13%(blanks[:(rowlen-len(row13)+3)])
		row14 = 'Days Flat:%s%0.2f%%%%'%('%s',100.0*days_flat/trade_days)
		row14 = row14%(blanks[:(rowlen-len(row14)+3)])
		row15 = 'Days Long:%s%0.2f%%%%'%('%s',100.0*days_long/trade_days)
		row15 = row15%(blanks[:(rowlen-len(row15)+3)])
		row16 = 'Days Short:%s%0.2f%%%%'%('%s',100.0*days_short/trade_days)
		row16 = row16%(blanks[:(rowlen-len(row16)+3)])
		row17 = 'Total Correlation:%s%0.4f'%('%s',correl)
		row17 = row17%(blanks[:(rowlen-len(row17)+2)])
		row18 = 'Up Days Correlation:%s%0.4f'%('%s',up_correl)
		row18 = row18%(blanks[:(rowlen-len(row18)+2)])
		row19 = 'Down Days Correlation:%s%0.4f'%('%s',dn_correl)
		row19 = row19%(blanks[:(rowlen-len(row19)+2)])
		row20 = 'Uncompounded Up Capture:%s%0.2f%%%%'%('%s',uncomp_up_capture*100)
		row20 = row20%(blanks[:(rowlen-len(row20)+3)])		
		row21 = 'Uncompounded Down Capture:%s%0.2f%%%%'%('%s',uncomp_dn_capture*100)
		row21 = row21%(blanks[:(rowlen-len(row21)+3)])
		# for this last section, break it into 40%, 30%, 40%
		col1len = int(1.0/3*rowlen); col2len = col1len; col3len = col2len
		rowCPH = '%sModel%sIndex%s'%(blanks[:col1len],blanks[:(col2len-5)],blanks[:(col3len-5)])		
		row22a = 'Up Days:%s'%('%s'); row22a = row22a%(blanks[:(col1len-len(row22a)+2)])
		row22b = '%d (%0.2f%%%%)%s'%(days_model_up,100.0*days_model_up/trade_days,'%s'); row22b = row22b%(blanks[:(col2len-len(row22b)+3)])
		row22c = '%d (%0.2f%%%%)%s'%(days_target_up,100.0*days_target_up/trade_days,'%s'); row22c = row22c%(blanks[:(col2len-len(row22c)+3)])
		row23a = 'Average Gain:%s'%('%s'); row23a = row23a%(blanks[:(col1len-len(row23a)+2)])
		row23b = '%0.2f%%%%%s'%(avg_model_up*100,'%s'); row23b = row23b%(blanks[:(col2len-len(row23b)+3)])
		row23c = '%0.2f%%%%%s'%(avg_target_up*100,'%s'); row23c = row23c%(blanks[:(col2len-len(row23c)+3)])
		row24a = 'Down Days:%s'%('%s'); row24a = row24a%(blanks[:(col1len-len(row24a)+2)])
		row24b = '%d (%0.2f%%%%)%s'%(days_model_dn,100.0*days_model_dn/trade_days,'%s'); row24b = row24b%(blanks[:(col2len-len(row24b)+3)])
		row24c = '%d (%0.2f%%%%)%s'%(days_target_dn,100.0*days_target_dn/trade_days,'%s'); row24c = row24c%(blanks[:(col2len-len(row24c)+3)])
		row25a = 'Average Loss:%s'%('%s'); row25a = row25a%(blanks[:(col1len-len(row25a)+2)])
		row25b = '%0.2f%%%%%s'%(avg_model_dn*-100,'%s'); row25b = row25b%(blanks[:(col2len-len(row25b)+3)])
		row25c = '%0.2f%%%%%s'%(avg_target_dn*-100,'%s'); row25c = row25c%(blanks[:(col2len-len(row25c)+3)])
		rowstss = '\tModel Stress Parameters:\n\tUp Moves Smoothed = %0.2f%%\n\tDown Moves Smoothed = %0.2f%%\n\tBest Signals Reversed = %0.2f%%'%(100*TopRemovePerc,100*BotRemovePerc,100*SigRemovePerc)
		
		# finally put them all together
		thePOOP = '\n'.join((thismodel,lins,row1,lins,row2,row3,row4,lins,rowDDH,lins,rowDDD,lins,rowDDs[1:],lins,\
		row5,row6,row7,row8,row9,row10,row10a,row11,lins,row12,row12a,row13,row14,row15,row16,lins,row17,row18,row19,\
		row20,row21,lins,rowCPH,row22a+row22b+row22c,row23a+row23b+row23c,row24a+row24b+row24c,\
		row25a+row25b+row25c,lins,rowstss,moddescrip))
		# diary output new JAH 20121031
		if saveoutput:
			# JAH 20140118 changed to calculate out_path
			out_path = __builtins__['QREDIS_out']+'/'+this_model.name[:this_model.name.index(str(this_model.model_id))]+'/'
			save_file = '%sQMPOOP_%s_%s%s.txt'%(out_path,this_model.name,['STR_',''][StressParams is None],QB.TimeStamp())
			QB.Diary(save_file)
		# I know I know I know; I could just print all that catenation stuff without wasting a variable
		# but I really wanted to say "print the POOP"!
		print(thePOOP)
		if saveoutput:
			print('POOP sheet saved in %s'%save_file)
			QB.Diary(None)
		else:
			save_file = None
	
	return model_stats, stat_names, DDstats, save_file

def TradePlot(this_model, trade_dates, model_return, target, long_return=None, short_return=None, out_file=None,logplot=False):
	"""
	Take the signals from a model over a specified time horizon and create a set of
	---
	Usage: fig = TradePlot(this_model, trade_dates, model_return, target, long_return, short_return, out_file, logplot)
	---
	this_model: this will be assumed to be an initialized model
	trade_dates: actual dates (from date + 1 : to date inclusive) for trades
	model_return: array_like of accumulated daily trade returns for the total model
	target: array_like of accumulated daily buy-and-hold returns for target index
	long_return: array_like of accumulated daily trade returns for the long signals only
	short_return: array_like of accumulated daily trade returns for the short signals only
	out_file*: string full path & file name (sans .eps) for saving plot to disk
		should go in QREDIS_out / model-specific folder; not saved if None
	logplot: create a plot with the x-axis: True = log-scaled, False*=linearly-scaled
	---
	ex: 
	JAH 20121011
	"""
	
	# threshhold for day-mon vs mon-year printing; up to 2 trading months will show daily
	nthresh = 40
	
	# in theory, this is taking the arrays all from TradeHistory, but check things anyway
	# duck the array_likes
	trade_dates = np.array(trade_dates, copy=False)
	model_return = np.array(model_return, copy=False)
	target = np.array(target, copy=False)
	long_return = np.array(long_return, copy=False)
	short_return = np.array(short_return, copy=False)
	# get the lengths
	try:
		lls = [len(trade_dates)]*5							# prefill with dates length
		lls[1] = len(model_return); lls[2] = len(target)	# get model and target lengths
	except TypeError:
		# if error here that means something wrong with most important arrays
		raise TypeError("Something wrong with dates, model, or target array(s): %s"%TradePlot.__doc__)
	# now fill in the optional arrays if appropriate
	if long_return is not None:
		lls[3] = len(long_return)
	if short_return is not None:
		lls[4] = len(short_return)
	
	# check all same length, and out_file parm
	if not(all([x == lls[0] for x in lls])):
		raise ValueError("Arrays should all be same length: %s"%TradePlot.__doc__)
	if (type(out_file) is not str) and (out_file is not None):
		raise TypeError("Variable out_file must either be None or full path/filename: %s"%TradePlot.__doc__)
	
	# get the x values
	n = lls[0]
	x = np.arange(n)
	
	# prepare the dates - store a list of string representations AND arrays of month/year numbers
	# JAH 20121106 added years array
	dates_string = np.array(['           ']*n,dtype=str)
	if n > nthresh:
		dates_mos = np.zeros(n,dtype=int); dates_yrs = dates_mos.copy()
		for dat in range(n):
			dates_string[dat] = trade_dates[dat].strftime('%d-%b-%Y')
			dates_mos[dat] = trade_dates[dat].month
			dates_yrs[dat] = trade_dates[dat].year
	else:
		for dat in range(n):
			dates_string[dat] = trade_dates[dat].strftime('%d-%m-%y')
	
	# build the plot
	fh = plt.figure(figsize = [10,6]); ax = plt.subplot(1,1,1)
	if logplot:
		plt.semilogy(x,model_return,'b',x,target,'k')
		plt.hold(True)
		if long_return is not None:
			plt.semilogy(x,long_return,'g--')
		if short_return is not None:
			plt.semilogy(x,short_return,'r--')	
	else:
		plt.plot(x,model_return,'b',x,target,'k')
		plt.hold(True)
		if long_return is not None:
			plt.plot(x,long_return,'g--')
		if short_return is not None:
			plt.plot(x,short_return,'r--')
	plt.hold(False)
	
	# now we want to show the calendar x-axis labels
	if n <= nthresh:
		# now set the ticks and tick labels; the labels are just the dd-mm part of the date strings
		ax.set_xticks(x)
		ax.set_xticklabels([dat[:5] for dat in dates_string],rotation=-45)
	else:			# otherwise just show months
		# first we find where the months change and get the indices
		newmon = x[dates_mos != np.roll(dates_mos,1,axis=0)]
		newyer = x[dates_yrs != np.roll(dates_yrs,1,axis=0)]
		# now set the ticks and labels to just these indices and just the mm-yy part of the date strings
		# only print the yyyy at the start of each year JAH 20121106
		ax.set_xticks(newmon)
		labs =  np.array([(dates_string[cnt][3]*(cnt in newmon))+\
			('\n'+dates_string[cnt][7:]*(cnt in newyer)) for cnt in range(n)])
		ax.set_xticklabels(labs[newmon])
	
	# finish up
	plt.title('Trading %s from %s - %s'%(this_model.target_index,trade_dates[0],trade_dates[-1]))
	plt.legend(('%s'%(this_model.name),'%s'%(this_model.target_index)),loc=2)
	plt.ylabel('Cumulative Compound Return')
	# write on the plot the compounded gain in green (loss in red)
	ax.annotate('%0.2f%%'%((model_return[-1]-1)*100),(x[-1],model_return[-1]),xytext=(10,10),\
		textcoords='offset points',color='rg'[model_return[-1]>=1])
	ax.annotate('%0.2f%%'%((target[-1]-1)*100),(x[-1],target[-1]),xytext=(10,10),textcoords='offset points')
	
	plt.show()
	if out_file is not None:
		plt.savefig(out_file+'.eps')
	
	return fh
	
def ModelEval(this_model, from_date=None, to_date=None, StressParams = None, saveoutput = False):
	"""
	This sequentially runs TradeHistory, POOP, and TradePlot with the most common and/or
	important parameters.
	---
	Usage: outs = ModelEval(this_model, from_date, to_date, StressParams, saveoutput)
	---
	this_model: this will be assumed to be an initialized model having at least the target_index attribute
	from_date* / to_date*: datetime.date first / last signal day; if None is 
		passed for either, the entire signal history is used
	StressParams: array_like holding 3 values: % of Top Up Days to smooth,
		% of Top Down Days to smooth, % of best trades to reverse or nullify
	saveoutput: True: save all files to QREDIS_mod / model-specific folder; False*: don't
	outs = 3-element tuple with the output from each function run
	---
	ex: 
	JAH 20121103
	"""
	
	# trade history
	dats, sigs, targrets, tot, lng, sht, targ, strss = TradeHistory(this_model, from_date, to_date, StressParams)
	
	# poop
	model_stats, stat_names, drawdowns, sv = TradePOOP(this_model, dats, sigs, \
	targrets, StressParams, POOPPrint =  True, saveoutput = saveoutput)
	
	if sv is not None:
		sv = sv[:-4]
		
	# trade plot
	fig = TradePlot(this_model, dats, tot, targ, lng, sht, sv)
	
	return (dats, sigs, targrets, tot, lng, sht, targ, strss), (model_stats, stat_names, drawdowns), fig
	

def ThreshTrade(signals, target_returns, threshhold = 0.0):
	"""
	Simulate trading for an array of target returns, and an array of either signals, or
	an array of floats that are to be converted to signals if they are absolutely
	large than a float threshold.
	---
	Usage: daily_returns = ThreshTrade(signals, target_returns, threshhold)
	---
	signals: array_like of daily trading signals or floats to be converted
	target_returns: array_like same size as signals of the target's daily returns
		for the same days
	threshhold*: float value such that if abs(signal) > threshhold, signal is
		generated; default is 0 (all non-0 days traded)
	daily_returns: array holding the uncompounded daily returns from trading
	---
	ex: 
	JAH 20121104
	"""
	
	# duck type the signals and target returns
	signals = np.array(signals,dtype=float).flatten()
	target_returns = np.array(target_returns,dtype=float).flatten()
	
	# check lens match and also check threshhold
	if (len(signals) != len(target_returns)) or (type(threshhold) is not float):
		raise ValueError("Signals and target returns should be same length, and threshhold should be float: %s"\
		%ThreshTrade.__doc__)
	
	return target_returns*np.sign(signals)*(np.abs(signals) > threshhold)

def CheatCheck(this_model, trade_date):
	"""
	This function runs a model's training and trading methods three times, attempting to
	identify a change in training parameters and generated signal due to a change in
	a future price value. Based on the specified trade date, it creates a training block
	with the dates up to but excluding the trade date, then runs the training and 
	trading methods. These are then run after perturbing the price of the target_index
	on the trade date by doubling it, then again by halving it. If the generated model
	parameters and signals don't change, it suggests the model is not "cheating" by
	somehow looking ahead. Any saved parameters/signals on the specified trade date will be
	deleted. You will be asked to approve clearing parameters and signals multiple times.
	---
	Usage: matches, parameters, signals = CheatCheck(this_model, trade_date)
	---
	this_model: this will be assumed to be an initialized model having at least attributes
		target_index, training_days, trading_days, buffer_days, and methods Train() and Trade()
	trade_date: date for which the model signal will be generated; this is the date on which
		the target price will be perturbed
	matches: tuple with 2 booleans: parameters all matched, and signals all matched
	parameters: tuple with 3 sets of parameter value tuples: normal trade/train,
		train/trade with 2x price on trade date, train/trade with 1/2x price
	signals: list of all three generated signals: normal, double, half
	---
	ex: 
	JAH 20121127
	"""
	
	# ensure input date is correct
	if type(trade_date) is not dat.date:
		raise TypeError("Trade date must be datetime.date: %s"%CheatCheck.__doc__)
	
	# first check if there are parameters or signals on trade_date already, and clear them
	try:
		# these will raise ValueError if no params/signals on this date
		tmp = this_model.GetParams(trade_date)
		tmp = this_model.GetSignals(trade_date, trade_date)
		print('There are existing parameters on %s; these will be cleared!'%trade_date)
		tmp = this_model.ClearPS(trade_date,trade_date)
		if not(tmp[0]):
			raise Exception("CheatCheck can't run unless you allow it to clear the existing stored parameters and signals!"	)
	except ValueError:
		# if there are no params, don't care
		sys.exc_clear()

	# first we get the training & trading blocks
	# get the training+buffer days - go back extra 30 just to be safe
	tmp = trade_date + dat.timedelta(days=-30 - this_model.buffer_days - this_model.training_days)
	d,h = QD.GetTickerCalendar(this_model.target_index,tmp,trade_date,False)
	# now just return the last buffer+training dates+trade_date (excluding holidays of course)
	tmp = d[~h][-(this_model.buffer_days+this_model.training_days+1):]
	# separate these into overlapping training+trading blocks
	train_block = tmp[:-1]
	# not this is not a real trade block, just the buffer + trade date
	trade_block = tmp[-(1+this_model.buffer_days):]

	# update the target_index for this model
	old_target = this_model.target_index
	this_model.target_index = "_CHEAT"

	# first model train+trade: with trade_date = 2x
	tmp = QD.MakeCheatData(old_target, train_block, trade_date,2.0)
	this_model.Train(train_block,trade_block,42)
	params_doub = this_model.GetParams(trade_date)
	signal_doub = this_model.Trade(trade_block)
	tmp = this_model.ClearPS(trade_date,trade_date)
	if not(tmp[0]):
		raise Exception("CheatCheck can't run unless you allow it to clear the temporary stored parameters and signals!")

	# second model train+trade: with trade_date = 1/2x
	tmp = QD.MakeCheatData(old_target, train_block, trade_date,0.5)
	this_model.Train(train_block,trade_block,42)
	params_half = this_model.GetParams(trade_date)
	signal_half = this_model.Trade(trade_block)
	tmp = this_model.ClearPS(trade_date,trade_date)
	if not(tmp[0]):
		raise Exception("CheatCheck can't run unless you allow it to clear the temporary stored parameters and signals!")
	
	# last model train+trade: normal target_index
	this_model.target_index = old_target
	this_model.Train(train_block,trade_block,42)
	params_norm = this_model.GetParams(trade_date)
	signal_norm = this_model.Trade(trade_block)
	tmp = this_model.ClearPS(trade_date,trade_date)
	if not(tmp[0]):
		raise Exception("CheatCheck can't run unless you allow it to clear the temporary stored parameters and signals!")
	
	# now that we have the 3 sets of model parameters & signals, compare them to check for equality
	nd =  np.all(params_norm == params_doub)
	nh =  np.all(params_norm == params_half)
	dh =  np.all(params_doub == params_half)
	prms = (nd and nh and dh)
	sigs = (signal_norm == signal_doub) and (signal_norm == signal_half) and (signal_doub == signal_half)
	# talk a little
	if prms:
		print('The three sets of model parameters/results ARE identical!')
	else:
		print('The three sets of model parameters/results ARE NOT identical!')
	if sigs:
		print('The three trading signals ARE identical!')
	else:
		print('The three trading signals ARE NOT identical!')
	if prms and sigs:
		print('All model parameters/results and signals ARE identical, so model does not seem to be cheating!')
	else:
		print('All model parameters/results and signals ARE NOT identical, so model might be cheating!\n\tPlease check the training procedure!')
	
	return (prms, sigs), (params_norm, params_doub, params_half), [signal_norm, signal_doub, signal_half]


#JAH 20130408 this would be nice, but can't get QMod_Template in here?
def LoadModel(this_model):
	"""
	Load a model class file into QREDIS, specified by the full pathname
	---
	Usage: LoadModel(this_model)
	---
	this_model: string filename for the model file to load
	---
	ex: LoadModel(QREDIS_mod + '/QMod_Example.py')
	JAH 20130408
	"""
	
	try:
		exec(open(this_model).read())
	except IOError:
		raise IOError("%s does not exist, or can't be accessed!"%this_model)
