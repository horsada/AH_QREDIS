#QREDIS_Data
"""
QREDIS data access functions
----------------------
Calendar - 20121001 - return dates from the QREDIS calendar for a specified date range
CalRange - 20120929 - return the first and last dates in the QREDIS calendar
CloseDB - 20120914 - close an open database connection
CreateModel - 20121001 - create a model in the QREDIS database; only call from model class
DelParams - 20120929 - delete parameters for a specified model and date range
DelSignals - 20120929 - delete signals for a specified model and date range
DelTickerPrice - 20140101 - delete prices for a given ticker and date from the QREDIS database
DescribeModel - 20120929 - save a model description; only call from model class
EdtTickerPrice - 20140101 - edit prices for a given ticker and date in the QREDIS database
GetLastParamsDate - 20121122 - get the last trade date for which the database has model parameters
GetModel - 20120923 - get an existing model from the database
GetModels - 20121113 - get all the models that match certain criteria
GetParams - 20120929 - get existing parameters for a specified model and date range
GetSettings - 20121103 - get settings for a specified model
GetSignals - 201201002 - get existing signals for a specifified model and date range
GetTickerCalendar - 20121001 - get the trading calendar for a specified ticker and date range
GetTickerLastClose - 20121122 - get the last closing price and date for a ticker
GetTickerPrices_L - 20121016 - get prices for a list of tickers for a list of dates
GetTickerPrices_R - 20120914 - get prices for a list of tickers in a specified date range
GetTickers - 20121001 - get all the tickers from the QREDIS database
GetTickersPrices_L - 20121016 - get prices for a ticker for a list of dates
GetTickersPrices_R - 20120923 - get prices for a ticker in a specified date range
LoadCSVData - 20130404 - load price data from a file into the QREDIS database
LoadEODData - 20140111 - download and load EOD price data into the QREDIS database
LoadXU100Data - 20140108 - download and load XU100 price data into the QREDIS database
MakeCheatData - 20121126 - create a fake ticker for the model cheat test
OpenDB - 20120914 - open a connection to the QREDIS database
PutParams - 20120924 - save parameters for a specified model on specific dates
PutSettings - 20121103 - save or update settings for a specified model
PutSignal - 20120924 - save a specific models signal on a specified date
PutTickerPrice - 20140101 - insert prices for a given ticker and date into the QREDIS database
ResetMe - 20120919 - close any open database connections and clear the user
----------------------
JAH 20140117 everything has been tested and seems to be working fine with python 3
"""

import os
import sys
#import MySQLdb as mdb
import pymysql as mdb # JAH switched 20140103 since MySQLdb doesn't support python3
import getpass
import string
import numpy as np
import datetime as dat
### imports required by ghazalpasha's code
import urllib.parse
import urllib.request
import httplib2
import http.client
import re
from io import BytesIO
from zipfile import ZipFile
### imports required by ghazalpasha's code

database_name = "QREDIS_Data"
database_server = "localhost"	# JAH this will change in a multi-computer environment
database_conn = None
database_open = False			# database is currently open?
my_password = None
my_user = getpass.getuser()

# JAH added 20160205: this is used by LoadCSVData to properly warn all callers
# that there is a problem with the data to be loaded;
# currently used to flag new tickers and new dates
class DataError(Exception):
    pass

# general database usage functions
def OpenDB():
	"""
	Open a connection to the back-end database for QREDIS Data.
	---	
	Usage: OpenDB()
	---
	JAH 20120914
	"""

	global my_password; global database_conn; global database_server
	global my_user; global my_password; global database_name
	global database_open

	# ensure we have the user's database password
	if my_password is None:
		tmp = getpass.getpass("Please enter your database password: ")
		if tmp != "":
			my_password = tmp
		else:
			raise ValueError("Empty Password, Can't Open Database!")

	# try to connect to datbase
	try:
		database_conn = mdb.connect(database_server, my_user, my_password, database_name)
	except mdb.Error as e:
		database_open = 0
		raise mdb.Error("OpenDB Error %d: %s" % (e.args[0],e.args[1]))

	database_open = True
	return database_open

	
def CloseDB():
	"""
	Close an open database connection.
	---
	Usage: CloseDB()
	---
	JAH 20120914
	"""

	global database_open; global database_conn

	if database_open == True:
		database_conn.close()
		database_open = False
	return True

	
def ResetMe():
	"""
	Reset the user password.
	---
	Usage: ResetUser()
	---
	JAH 20120919
	"""

	global my_password
	try:
		CloseDB()
	except Exception:
		1/1
		# if not connected, do nothing - it doesn't really matter JAH 20121015
		# sys.exc_clear() JAH comment out for Python 3 20140111
	my_password = None
	return True
# general database usage functions

# ticker data functions
def GetTickerPrices_R(ticker, from_date=dat.date(1000,1,1), to_date=dat.date.today(), OHLCV = "Close"):
	"""
	Get prices in a date range for a specified ticker. If the dates are not passed, the
	entire history up through today is returned.
	---
	Usage: dates, holidays, prices = GetTickerPrices_R(ticker, from_date, to_date, OHLCV)
	---
	ticker: string indicating ticker to extract
	from_date* / to_date*: datetime date variables date range
	OHLCV: list variable including at least one of "Open", "High", "Low", "Close"*, "Volume"
	dates: array holding date(s) (as datetime.date)
	holidays: array holding holiday flags as boolean
	prices: array holding the price field(s) requested
	---
	ex: d,h,p = QD.GetTickerPrices_R(ticker="SP500", from_date=dat.date(2012,9,1), OHLCV=["Close","High"])
	JAH 20120914
	"""
 
	global database_open; global database_conn

	# duck type ticker, maybe it's np.string, by trying to uppercase it
	# this won't work if it's not a string-type variable JAH 20130320
	try:
		jnk = ticker.upper()
	except AttributeError:
		raise TypeError("Ticker must be a string: %s"%GetTickerPrices_R.__doc__)

	# ensure input variables are correct
	# JAH 20120923 changed dates to datetime.date (which apparently can't be cast from string)
	if (type(from_date) is not dat.date) or (type(to_date) is not dat.date):
		raise TypeError("Variable(s) are wrong type: %s"%GetTickerPrices_R.__doc__)

	# convert the OHLCV list to a string - must remove the quotes and brackets if a list
	if type(OHLCV) is not str:
		OHLCV = str(OHLCV)
		OHLCV = OHLCV[1:-1].replace("'","")
	# JAH added 20130321 putting the ifnull( ,0) around the fields
	OHLCV = 'ifnull('+OHLCV.replace(", ",",0), ifnull(")+',0)'

	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e

	# if it got here, db should be open JAH 20120923 changed query to use calendar
	cur = database_conn.cursor()
	# JAH changed to this 20130321 to ensure missing days are extracted
	res = cur.execute("select C.cal_date, C.holi, %s from (select T.Ticker, cal_date, locate(concat(T.holidays,';'),C.holidays)>0 as holi from tblCalendar AS C, "\
	"(SELECT ticker, holidays FROM tblIndex WHERE ticker = '%s') AS T where cal_date >= '%s' and cal_date <= '%s') as C LEFT JOIN tblIndexDaily as I ON C.ticker = I.ticker and C.cal_date = I.price_date;"\
	%(OHLCV, ticker, from_date.isoformat(), to_date.isoformat()))
	
	# maybe there can be no data?
	if res == 0:
		cur.close()
		raise ValueError("No prices found for %s between %s and %s!"%(ticker,from_date,to_date))
	
	# get the data from the cursor and prepare output, separation code JAH 20120919
	tmp = cur.fetchall(); cur.close();
	dats = np.array([x[0] for x in tmp], dtype = dat.date)		# get the dates in one array ...
	holis = np.array([x[1] for x in tmp], dtype = bool)			# ... the holiday flags in another
	prics = np.array([x[2:] for x in tmp])						# ... and the prices in another

	return dats,holis,prics

def GetTickerPrices_L(ticker, dates_list, OHLCV = ["Close"]):
	"""
	Get prices for specific dates for a specified ticker.  An array of dates is passed
	as output also, since perhaps the user accidentally specifies weekends, which
	don't exist in the QREDIS calendar.
	---
	Usage: dates, holidays, prices = GetTickerPrices_R(ticker, dates_list, OHLCV)
	---
	ticker: string indicating ticker to extract
	dates_list: array_like of datetime dates indicating which dates to extract
	OHLCV: list variable including at least one of "Open", "High", "Low", "Close"*, "Volume"
	dates: array holding date(s) (as datetime.date) extracted
	holidays: array holding holiday flags as boolean
	prices: array holding the price field(s) requested
	---
	ex: d,h,p = QD.GetTickerPrices_L("SP500", [dat.date(2012,9,2),dat.date(2012,9,3),dat.date(2012,9,4),dat.date(2012,9,5)])
	JAH 20121016
	"""
 
	global database_open; global database_conn
	
	# duck type ticker, maybe it's np.string, by trying to uppercase it
	# this won't work if it's not a string-type variable JAH 20130320
	try:
		jnk = ticker.upper()
	except AttributeError:
		raise TypeError("Ticker must be a string: %s"%GetTickerPrices_L.__doc__)	

	# convert the OHLCV list to a string - must remove the quotes and brackets if a list
	if type(OHLCV) is not str:
		OHLCV = str(OHLCV)
		OHLCV = OHLCV[1:-1].replace("'","")
	# JAH added 20130321 putting the ifnull( ,0) around the fields
	OHLCV = 'ifnull('+OHLCV.replace(", ",",0), ifnull(")+',0)'
	
	# convert the dates array to a string, removing brackets and inserting quotes & commas
	#dats = str(dates_list); dats = string.replace(dats[1:-1],' ',"','")
	# JAH 20140111 changed because string changed in python3; can't think of how to do this without a loop :-(
	dats =  "'"
	for d in dates_list:
		dats += d.isoformat()+ "','"

	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e

	# if it got here, db should be open
	cur = database_conn.cursor()
	# JAH changed to this 20130321 to ensure missing days are extracted
	res = cur.execute("select C.cal_date, C.holi, %s from (select T.Ticker, cal_date, locate(concat(T.holidays,';'),C.holidays)>0 as holi from "\
	"tblCalendar AS C, (SELECT ticker, holidays FROM tblIndex WHERE ticker = '%s') AS T where cal_date IN (%s)) as C LEFT JOIN tblIndexDaily as I ON C.ticker = I.ticker and C.cal_date = I.price_date;"\
	%(OHLCV, ticker, dats[:-2]))

	# maybe there can be no data?
	if res == 0:
		cur.close()
		raise ValueError("No prices found for %s on dates [%s]!"%(ticker,dats))
	
	# get the data from the cursor and prepare output
	tmp = cur.fetchall(); cur.close();
	dats = np.array([x[0] for x in tmp], dtype = dat.date)		# get the dates in one array ...
	holis = np.array([x[1] for x in tmp], dtype = bool)			# ... the holiday flags in another
	prics = np.array([x[2:] for x in tmp])						# ... and the prices in another

	return dats,holis,prics

def GetTickersPrices_R(tickers, from_date=dat.date(1000,1,1), to_date=dat.date.today(), OHLCV = "Close"):
	"""
	Get prices for a specified set of tickers in a specified date range. If the dates
	are not passed, all prices through today are extracted.
	---
	Usage: dates, holidays, prices = GetTickersPrices_R(tickers, from_date, to_date, OHLCV)
	---
	tickers: list of strings indicating tickers to extract
	from_date* / to_date*: datetime date variables date range
	OHLCV: string with one of these "Open", "High", "Low", "Close"*, "Volume"
	dates: array holding date(s) (as datetime.date)
	holidays: array holding holiday flags as boolean
	prices: array holding the price field(s) requested
	---
	ex: d,h,p = QD.GetTickersPrices_R(["SP500","NDX"], dat.date(2012,9,1), dat.date(2012,9,30),"Close")
	JAH 20120923
	"""
 
	global database_open; global database_conn

	# ensure input variables are correct
	# can't duck type ticker or OHLCV because str-list or list-str adds to many complications JAH
	if (type(tickers) is not list) or (type(from_date) is not dat.date)\
		or (type(to_date) is not dat.date) or (type(OHLCV) is not str):
		raise TypeError("Variable(s) are wrong type: %s"%GetTickersPrices_R.__doc__)

	# convert the ticker list to a string - must remove the brackets
	tmp = str(tickers)[1:-1]

	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e

	# if it got here, db should be open
	cur = database_conn.cursor()
	# JAH changed to this 20130321 to ensure missing days are extracted
	res = cur.execute("select C.ticker, C.cal_date, C.holi, ifnull(%s,0) from (select T.Ticker, cal_date, locate(concat(T.holidays,';'),C.holidays)>0 as holi from "\
	"tblCalendar AS C, (SELECT ticker, holidays FROM tblIndex WHERE ticker IN (%s)) AS T where cal_date >= '%s' and cal_date <= '%s') as C LEFT JOIN "\
	"tblIndexDaily as I ON C.ticker = I.ticker and C.cal_date = I.price_date;"\
	%(OHLCV, tmp, from_date.isoformat(), to_date.isoformat()))
	
	# maybe there can be no data?
	if res == 0:
		cur.close()
		raise ValueError("No prices found for %s between %s and %s!"%(tmp,from_date,to_date))

	# get the data from the cursor and prepare output
	tmp = cur.fetchall(); cur.close()
	
	# separate the columns of data: tickers, dates, prices
	tick_names = np.array([x[0] for x in tmp],dtype=str)			# get ticker names
	dats = np.array([x[1] for x in tmp], dtype = dat.date)		# get holidays
	hols = np.array([x[2] for x in tmp], dtype = bool)			# get dates
	prics = np.array([x[-1] for x in tmp])						# get prices

	# "transpose" the prices/holidays vectors so each column is a ticker's prices/holidays
	prices = np.zeros((len(tickers),len(prics)/len(tickers)),dtype=float)
	holis = np.zeros((len(tickers),len(prics)/len(tickers)),dtype=bool)
	for cnt in range(len(tickers)):
		tick = tickers[cnt]
		prices[cnt,:] = prics[tick_names==tick]
		holis[cnt,:] = hols[tick_names==tick]
	
	# now make the dates vector show every date only once
	dats = dats[tick_names==tickers[0]]
	
	return dats,holis.T,prices.T
	
def GetTickersPrices_L(tickers, dates_list, OHLCV = "Close"):
	"""
	Get prices for specific dates for a specified set of tickers.  An array of
	dates is passed as output also, since perhaps the user accidentally specifies
	weekends, which don't exist in the QREDIS calendar.
	---
	Usage: dates, holidays, prices = GetTickersPrices_L(tickers, dates_list, OHLCV)
	---
	tickers: list of strings indicating tickers to extract
	dates_list: array_like of datetime dates indicating which dates to extract
	OHLCV: string with one of these "Open", "High", "Low", "Close"*, "Volume"
	dates: array holding date(s) (as datetime.date)
	holidays: array holding holiday flags as boolean
	prices: array holding the price field(s) requested
	---
	ex: d,h,p = QD.GetTickersPrices_L(["SP500","NDX"], [dat.date(2012,9,2),dat.date(2012,9,3),dat.date(2012,9,4),dat.date(2012,9,5)])
	JAH 20121016
	"""
 
	global database_open; global database_conn

	# ensure input variables are correct
	if (type(tickers) is not list) or (type(OHLCV) is not str):
		raise TypeError("Variable(s) are wrong type: %s"%GetTickersPrices_L.__doc__)

	# convert the ticker list to a string - must remove the brackets
	tmp = str(tickers)[1:-1]
	# convert the dates array to a string, removing brackets and inserting quotes & commas
	#dats = str(dates_list); dats = string.replace(dats[1:-1],' ',"','")
	# JAH 20140111 changed because string changed in python3; can't think of how to do this without a loop :-(
	dats =  "'"
	for d in dates_list:
		dats += d.isoformat()+ "','"

	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e

	# if it got here, db should be open
	cur = database_conn.cursor()
	# JAH changed to this 20130321 to ensure missing days are extracted
	res = cur.execute("select C.ticker, C.cal_date, C.holi, ifnull(%s,0) from (select T.Ticker, cal_date, locate(concat(T.holidays,';'),"\
	"C.holidays)>0 as holi from tblCalendar AS C, (SELECT ticker, holidays FROM tblIndex WHERE ticker IN (%s)) AS T where cal_date IN (%s)) as C "\
	"LEFT JOIN tblIndexDaily as I ON C.ticker = I.ticker and C.cal_date = I.price_date;"\
	%(OHLCV, tmp, dats[:-2]))

	# maybe there can be no data?
	if res == 0:
		cur.close()
		raise ValueError("No prices found for %s on dates [%s]!"%(tmp,dats))

	# get the data from the cursor and prepare output
	tmp = cur.fetchall(); cur.close()
	
	# separate the columns of data: tickers, dates, holidays, prices
	tick_names = np.array([x[0] for x in tmp],dtype=str)			# get ticker names
	dats = np.array([x[1] for x in tmp], dtype = dat.date)		# get holidays
	hols = np.array([x[2] for x in tmp], dtype = bool)			# get dates
	prics = np.array([x[-1] for x in tmp])						# get prices

	# "transpose" the prices/holidays vectors so each column is a ticker's prices/holidays
	prices = np.zeros((len(tickers),len(prics)/len(tickers)),dtype=float)
	holis = np.zeros((len(tickers),len(prics)/len(tickers)),dtype=bool)
	for cnt in range(len(tickers)):
		tick = tickers[cnt]
		prices[cnt,:] = prics[tick_names==tick]
		holis[cnt,:] = hols[tick_names==tick]
	
	# now make the dates vector show every date only once
	dats = dats[tick_names==tickers[0]]
	
	return dats,holis.T,prices.T
	
def GetTickerCalendar(ticker, from_date=dat.date(1000,1,1), to_date=dat.date.today(), trim=False):
	"""
	Get the trading calendar in a specified date range for a specific ticker.  This returns the
	entire calendar in the date range + an array of holiday flags.  The ticker calendar can then be
	obtained by dates[~holidays].  If the dates are not passed, it will return everything, up
	through today.
	---
	Usage: dates, holidays = GetTickerCalendar(ticker, from_date, to_date, trim)
	---
	ticker: string indicating ticker to extract
	from_date* / to_date*: datetime date variables date range
	trim*: if True, the front is trimmed to only include dates after prices start,
		and the back is trimmed after the last price
	dates: datetime.date array holding ALL dates
	holidays: bool array same size as dates holding holiday flags
	---
	ex: d,h = QD.GetTickerCalendar("SP500", dat.date(2012,1,1), dat.date(2012,9,30))
	JAH 20121001
	"""
 
	global database_open; global database_conn

	# duck type ticker, maybe it's np.string, by trying to uppercase it
	# this won't work if it's not a string-type variable JAH 20130320
	try:
		jnk = ticker.upper()
	except AttributeError:
		raise TypeError("Ticker must be a string: %s"%GetTickerCalendar.__doc__)

	# ensure input variables are correct
	if (type(from_date) is not dat.date) or (type(to_date) is not dat.date):
		raise TypeError("Dates should be dat.date: %s"%GetTickerCalendar.__doc__)

	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e
	
	# if it got here, db should be open
	cur = database_conn.cursor()
	# if user wants the calendar trimmed, first get the min and max JAH 20121122
	if trim:
		res = cur.execute("SELECT min(price_date), max(price_date) FROM tblIndexDaily where ticker = '%s' AND price_date >= '%s' and price_date <= '%s' and close > 0;"\
		%(ticker, from_date.isoformat(), to_date.isoformat()))
		tmp = cur.fetchone();
		# maybe there can be no data; if so, the min & max will be NULL=None
		if tmp[0]==None:
			cur.close()
			raise ValueError("No data with prices found for %s, or no calendar dates between %s and %s!"%(ticker,from_date.isoformat(),to_date.isoformat()))
		(from_date,to_date) = tmp
	# now we can get the (possibly trimmed) calendar
	res = cur.execute("SELECT C.cal_date, locate(concat(I.holidays,';'),C.holidays)>0 FROM tblIndex AS I, tblCalendar AS C WHERE I.ticker = '%s' AND C.cal_date >= '%s' and C.cal_date <= '%s';"\
	%(ticker, from_date.isoformat(), to_date.isoformat()))
	
	# maybe there can be no data?
	if res == 0:
		cur.close()
		raise ValueError("No data found for %s, or no calendar dates between %s and %s!"%(ticker,from_date,to_date))
	
	# get the data from the cursor and prepare output, separation code JAH 20120919
	tmp = cur.fetchall(); cur.close();
	dats = np.array([x[0] for x in tmp], dtype = dat.date)		# get the dates in one array ...
	holis = np.array([x[1] for x in tmp], dtype = bool)			# ... the holiday flags in another

	return dats,holis

def GetTickerLastClose(ticker):
	"""
	Get the last closing price in the QREDIS database for a specific ticker.
	---
	Usage: date, close = GetTickerLastClose(ticker)
	---
	ticker: string indicating ticker to extract
	dates: datetime.date of the last closing price
	close: last closing price in the database
	---
	ex: d,p = QD.GetTickerLastClose("SP500")
	JAH 20121122
	"""
 
	global database_open; global database_conn

	# duck type ticker, maybe it's np.string, by trying to uppercase it
	# this won't work if it's not a string-type variable JAH 20130320
	try:
		jnk = ticker.upper()
	except AttributeError:
		raise TypeError("Ticker must be a string: %s"%GetTickerLastClose.__doc__)

	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e

	# if it got here, db should be open
	cur = database_conn.cursor()
	res = cur.execute("SELECT price_date, close FROM tblIndexDaily WHERE ticker = '%s' and close > 0 ORDER BY price_date DESC LIMIT 1;"%ticker)

	# maybe there can be no data?
	if res == 0:
		cur.close()
		raise ValueError("No data with prices found for %s!"%ticker)
	
	# get the data from the cursor
	tmp = cur.fetchall(); cur.close();

	return tmp[0]
# ticker data functions

# model functions
def MakeCheatData(ticker, train_block, trade_date, trade_mult):
	"""
	Copy real ticker data to the fake _CHEAT ticker in the QREDIS database so it can
	be used to test a model for "cheating"
	---
	ex: THIS SHOULD ONLY BE CALLED FROM THE CheatCheck FUNCTION IN Qredis_Model!!!
	JAH 20121126
	"""
 
	global database_open; global database_conn
	
	# ducktype trade date multiplier
	try:
		trade_mult = float(trade_mult)
	except TypeError:
		raise TypeError("Trade date multiplier should be an integer or float: %s"%MakeCheatData.__doc__)
	# duck type ticker, maybe it's np.string, by trying to uppercase it
	# this won't work if it's not a string-type variable JAH 20130320
	try:
		jnk = ticker.upper()
	except AttributeError:
		raise TypeError("Ticker must be a string: %s"%MakeCheatData.__doc__)		

	# ensure input variables are correct
	if type(trade_date) is not dat.date:
		raise TypeError("Trade date should be a datetime.date object: %s"%MakeCheatData.__doc__)

	# convert the dates array to a string, removing brackets and inserting quotes & commas
	#dats = str(train_block); dats = string.replace(dats[1:-1],' ',"','")
	# JAH 20140111 changed because string changed in python3; can't think of how to do this without a loop :-(
	dats =  "'"
	for d in train_block:
		dats += d.isoformat()+ "','"

	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e

	# if it got here, db should be open
	cur = database_conn.cursor()
	# first we should delete any _CHEAT records with these dates
	res = cur.execute("DELETE FROM tblIndexDaily WHERE ticker = '_CHEAT' AND price_date IN (%s,'%s');"%(dats[:-2],trade_date.isoformat()))
	database_conn.commit()
	
	# now we can insert the training data
	res = cur.execute("INSERT INTO tblIndexDaily SELECT '_CHEAT',price_date,open,high,low,close,volume,Null FROM tblIndexDaily "\
	"WHERE ticker = '%s' AND price_date IN (%s) UNION SELECT '_CHEAT',price_date,open*%0.4f,high*%0.4f,low*%0.4f,close*%0.4f,volume*%0.4f,Null "\
	"FROM tblIndexDaily WHERE ticker = '%s' AND price_date = '%s';"\
	%(ticker,dats[:-2],trade_mult,trade_mult,trade_mult,trade_mult,trade_mult,ticker,trade_date.isoformat()))
	database_conn.commit()
	cur.close()
	
	# make sure records were insterted
	if res == 0:
		raise ValueError("Unable to create cheat test ticker!")
	
	return res


def CreateModel(modname, target, train, trade, buff, sdata):
	"""
	Create a new model in the database and get the id.
	---
	ex: THIS SHOULD ONLY BE CALLED FROM A MODEL CLASS!!!
	JAH 20120923
	"""

	global database_open; global database_conn; global my_user
	
	# ensure input variables are correct
	if (type(modname) is not str) or (type(target) is not str) or (type(train) is not int) or \
		(type(trade) is not int) or (type(buff) is not int) or (type(sdata) is not str):
		raise TypeError("Variable(s) are wrong type: %s"%__CreateModel.__doc__)

	# if there are any, replace the 's with "s, then we can take this from the db later
	# and easily undo it JAH 20130526
	# JAH 20140108 changed to accomodate python 3
	#sdata = string.replace(sdata,"'",'"')
	sdata = sdata.replace("'",'"')

	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e
	
	# if it got here, db should be open
	# first we insert the new model
	cur = database_conn.cursor()
	res = cur.execute("INSERT INTO tblModel(model_name,target_index,training_days,trading_days,buffer_days,source_data,create_user) VALUES('%s','%s',%d,%d,%d,'%s','%s');"\
	%(modname, target, train, trade, buff, sdata, my_user))
	database_conn.commit()
	
	# make sure 1 record was insterted
	if res == 0:
		cur.close()
		raise ValueError("Unable to create model ('%s','%s',%d,%d,%d,'%s')"%(modname, target, train, trade, buff, sdata))	
	
	# now we get the id
	res = cur.execute("SELECT max(model_id) FROM tblModel WHERE create_user = '%s' GROUP BY create_user;"%my_user)
	
	# make sure it got a record
	if res == 0:
		cur.close()
		raise ValueError("Unable to get model ('%s','%s',%d,%d,%d'%s')"%(modname, target, train, trade, buff, sdata))

	# get the data from the cursor
	tmp = cur.fetchone(); cur.close()
	
	#return int(tmp[0]) # JAH 20120924 do I really have to cast this? JAH 20140104 changed for python3
	return tmp[0]


def GetModel(model_id):
	"""
	Get model definition from the database.
	---
	Usage: model name, target index, training days, trading days, buffer days, source data = GetModel(model_id)
	---
	model_id: numerical id of model in database
	descrip: string description of the model
	target index: ticker of index this model trades
	training / trading / buffer days: number of days used for each training and trading, plus the buffer for both
	source data: string describing source data maybe a string'd list of tickers
	model name: name of the model class as initially defined
	---
	ex: 
	JAH 20120923
	"""

	global database_open; global database_conn

	# ensure input variables are correct
	if type(model_id) is not int:
		raise TypeError("Model id should be integer: %s"%GetModel.__doc__)

	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e

	# if it got here, db should be open
	cur = database_conn.cursor()
	res = cur.execute("SELECT model_descrip, target_index, training_days, trading_days, buffer_days, source_data, model_name FROM tblModel WHERE model_id = %d;"%(model_id))
	
	# maybe there can be no data
	if res == 0:
		cur.close()
		raise ValueError("No model %d found!"%(model_id))
	
	# get the data from the cursor
	tmp = cur.fetchone(); cur.close()
	
	# JAH 20130526 when we stored the source_data string, ' was converted to ",
	# so undo it
	sdata =  str.replace(tmp[5],'"',"'")	
	
	# have to cast the training & trading days to int, since they're inexplicably long in db
	# JAH 20120924 do I really have to cast this? JAH 20140104 changed for python3
	#return (tmp[0],tmp[1],int(tmp[2]),int(tmp[3]),int(tmp[4]),sdata,tmp[-1]);
	return (tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],sdata,tmp[-1])
	
	
def PutParams(model_id, param_dates, param_names, param_values, param_types):
	"""
	Save parameters from a model run for a set of specific training dates into the database.
	Parameter values will be stored as strings (up to 200 characters).
	---
	Usage: result = PutParams(model_id, param_dates, param_names, param_values, param_types)
	---
	model_id: numerical id of model in database
	param_dates: datetime.date list or array of trading days for which the parms should be used
	param_names: array_like of parameter names to be stored as strings(20); no special
		characters, no spaces, nothing that might be interpreted!
	param_values: array_like of values as same size, and in the same order, as names
	param_types: array_like of python variable types stored as strings(10) as same size, and in
		the same order, as names; these can be used to convert from string back
	result: number parameters inserted into the database
	---
	ex: THIS SHOULD ONLY BE CALLED FROM A MODEL CLASS!!!
	JAH 20120924
	"""
	
	global database_open; global database_conn; global my_user
	
	# ducktype dates to datetime.dat array
	param_dates = np.array(param_dates, dtype = dat.date, ndmin = 1)

	# ensure input variables are correct
	if (type(model_id) is not int):
		raise TypeError("Variable(s) are wrong type: %s"%PutParams.__doc__)
	if (len(param_names) != len(param_values)) or (len(param_names) != len(param_types)):
		raise ValueError("Number of parameter names and values must match: %s"%PutParams.__doc__)

	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e
	
	# if it got here, db should be open
	# WTF! mysqldb connector package does not support multiple statements in a single execute WTF!
	# but we get around this by building a UNION query and inserting that JAH 20120925 :-) daaaamn!
	
	# build the SQL first - JAH 20121002 changed to store param_types
	# first we build a template with the id, today's date, and user
	SQLinsertT = "UNION ALL SELECT %s AS model_id, '%%s' as param_date, '%%s' as param_name, '%%s' as param_value, '%%s' as param_type, '%s' as run_date, '%s' as run_user "\
	%(model_id, dat.date.today().isoformat(), my_user)
	# now broadcast each name-value pair into the template, then catenate them all together to form the sql script
	# this is nested within the list comprehension for date so each date is broadcast to each name-value pair
	SQLinserts = " ".join([" ".join([SQLinsertT%(d.isoformat(),n,v,f) for n,v,f in zip(param_names,param_values,param_types)]) for d in param_dates])

	cur = database_conn.cursor()	
	res = cur.execute("INSERT INTO tblModelParam "+SQLinserts[10:-1]+";")
	database_conn.commit()
	cur.close()
	
	return int(res)


def PutSignal(model_id, sig_date, signal, backtest):
	"""
	Save the signal from a model, generated for a specific date.
	---
	Usage: result = PutSignal(model_id, sig_date, signal, backtest)
	---
	model_id: numerical id of model in database
	sig_date: datetime.date date for which the signal was generated
	signal: integer signal 1 = buy long, -1 = sell short
	backtest: this comes automatically from the QREDIS_Model module
	result: True if signal was inserted into the dabase
	---
	ex: THIS SHOULD ONLY BE CALLED FROM A MODEL CLASS!!!
	JAH 20120924
	"""
	
	global database_open; global database_conn; global my_user
	
	# maybe signal can be a list/tuple/array, so try to convert (while casting to int)
	try:
		signal = int(signal[0])
	except Exception:
		# do nothing, just let it slide, but go ahead and cast as int
		signal = int(signal)

	# ensure input variables are correct
	if (type(model_id) is not int) or (type(sig_date) is not dat.date):
		raise TypeError("Variable(s) are wrong type: %s"%PutSignal.__doc__)
	if abs(signal) > 1:
		raise ValueError("Signal must be 1, 0, or -1: %s"%PutSignal.__doc__)

	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e

	# if it got here, db should be open
	cur = database_conn.cursor()
	res = cur.execute("INSERT INTO tblModelSignal Values(%d,'%s',%d, '%s','%s','%s');"\
	%(model_id, sig_date.isoformat(), signal, backtest, dat.date.today().isoformat(), my_user))
	database_conn.commit()
	cur.close()

	return (res>0);


def GetSignals(model_id, from_date=dat.date(1000,1,1), to_date=dat.date.today()):
	"""
	Get a model's stored signals over a specific time horizon. If the dates are not
		passed, all signals up through today are returned.
	---
	Usage: signal dates,signals = GetSignals(model_id, from_date, to_date)
	---
	model_id: numerical id of model in database
	from_date* / to_date*: datetime.date variables date range to get the signals
	signals: array of model signals
	signal dates: datetime.date array indicating the date on which the signals are
		generated; they should be traded the next day
	---
	ex: 
	JAH 201201002
	"""

	global database_open; global database_conn
	
	# ensure input variables are correct
	if (type(model_id) is not int) or (type(from_date) is not dat.date) or (type(to_date) is not dat.date):
		raise TypeError("Model id should be integer, and dates should be datetime.date: %s"%GetSignals.__doc__)
	
	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e
			
	# if it got here, db should be open
	cur = database_conn.cursor()
	res = cur.execute("SELECT signal_date, signal_value FROM tblModelSignal WHERE model_id = %d AND signal_date >= '%s' AND signal_date <= '%s';"\
	%(model_id, from_date.isoformat(), to_date.isoformat()))
	
	# maybe there can be no data
	if res == 0:
		cur.close()
		raise ValueError("No signals for model %d between %s and %s found!"%(model_id,from_date,to_date))
		
	tmp = cur.fetchall(); cur.close()
	
	# extract the variables
	signal_dates = np.array([x[0] for x in tmp],dtype=dat.date)		# get signal dates
	signal_values = np.array([x[1] for x in tmp],dtype=int)			# get signal values

	return signal_dates, signal_values
	
	
def GetLastParamsDate(model_id):
	"""
	Get the last date with parameters in the QREDIS database for a specified model.
	---
	Usage: param_date = GetLastParamsDate(model_id)
	---
	model_id: numerical id of model in database
	param_date: datetime.date day of the last set of parameters
	---
	ex: 
	JAH 20121122
	"""

	global database_open; global database_conn

	# ensure input variables are correct
	if type(model_id) is not int:
		raise TypeError("Model id should be integer: %s"%GetLastParamsDate.__doc__)
	
	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e
			
	# if it got here, db should be open
	cur = database_conn.cursor()
	res = cur.execute("SELECT max(param_date) FROM tblModelParam WHERE model_id = %d;"%model_id)
	
	# maybe there can be no data
	if res == 0:
		cur.close()
		raise ValueError("No parameters for model %d found!"%model_id)
		
	tmp = cur.fetchone(); cur.close()
	
	return tmp[0]


def GetParams(model_id, param_date):
	"""
	Get a model's stored parameters for a specific date.
	---
	Usage: names, values, types = GetParams(model_id, param_date)
	---
	model_id: numerical id of model in database
	param_date: datetime.date day for which the parameters should be extracted
	names: array of parameter names
	values: array of values stored as strings as same size, and in the same order, as names
	types: array of python variable types stored as strings as same size, and in
		the same order, as names; these can be used to convert from string back
	---
	ex: 
	JAH 20120929
	"""

	global database_open; global database_conn

	# ensure input variables are correct
	if (type(model_id) is not int) or (type(param_date) is not dat.date):
		raise TypeError("Model id should be integer, and date should be datetime.date: %s"%GetParams.__doc__)
	
	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e
			
	# if it got here, db should be open
	cur = database_conn.cursor()
	res = cur.execute("SELECT param_name, param_value, param_type FROM tblModelParam WHERE model_id = %d AND param_date = '%s';"%(model_id, param_date.isoformat()))
	
	# maybe there can be no data
	if res == 0:
		cur.close()
		raise ValueError("No parameters for model %d on %s found!"%(model_id,param_date))
		
	tmp = cur.fetchall(); cur.close()
	
	# extract the variables
	pnams = np.array([x[0] for x in tmp],dtype=str)		# get parameter names
	pvals = np.array([x[1] for x in tmp],dtype=str)		# get parameter values
	ptyps = np.array([x[2] for x in tmp],dtype=str)		# get parameter types

	return pnams,pvals,ptyps
	
	
def DescribeModel(model_id,descrip):
	"""
	Store the description (up to 500 characters) for a model.
	---
	ex: THIS SHOULD ONLY BE CALLED FROM A MODEL CLASS!!!
	JAH 20120929
	"""	
	
	global database_open; global database_conn;

	# ensure input variables are correct
	if type(model_id) is not int:
		raise TypeError("Model ID should be int: %s"%__DescribeModel.__doc__)

	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e

	# if it got here, db should be open
	cur = database_conn.cursor()
	res = cur.execute("Update tblModel SET model_descrip = '%s' WHERE model_id = %d"%(descrip[:501],model_id))
	database_conn.commit()
	cur.close()
	
	return	(res>0)
	
	
def DelSignals(model_id, from_date=dat.date(1000,1,1), to_date=dat.date(9999,12,31)):
	"""
	Clear a model's signals in a specified date range.  If dates are not passed, all
		signals are cleared.
	---
	Usage: result = DelSignals(model_id, from_date, to_date)
	---
	model_id: numerical id of model in database
	from_date* / to_date*: datetime date variables date range
	result: number of signals deleted from the database
	---
	ex: 
	JAH 20120929
	"""
	
	global database_open; global database_conn

	# ensure input variables are correct
	if (type(model_id) is not int) or (type(from_date) is not dat.date) or (type(to_date) is not dat.date):
		raise TypeError("Variable(s) are wrong type: %s"%DelSignals.__doc__)

	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e	
	
	# if it got here, db should be open
	cur = database_conn.cursor()
	res = cur.execute("DELETE FROM tblModelSignal WHERE model_id = %d AND signal_date >= '%s' AND signal_date <= '%s';"\
	%(model_id,from_date.isoformat(),to_date.isoformat()))
	database_conn.commit()
	cur.close()	
	
	return res
	
	
def DelParams(model_id, from_date=dat.date(1000,1,1), to_date=dat.date(9999,12,31)):
	"""
	Clear a model's parameters in a specified date range. If dates are not passed, all
		signals are cleared.
	---
	Usage: result = DelParams(model_id, from_date, to_date)
	---
	model_id: numerical id of model in database
	from_date* / to_date*: datetime date variables date range
	result: number of parameters deleted from the database
	---
	ex: 
	JAH 20120929
	"""
	
	global database_open; global database_conn

	# ensure input variables are correct
	if (type(model_id) is not int) or (type(from_date) is not dat.date) or (type(to_date) is not dat.date):
		raise TypeError("Variable(s) are wrong type: %s"%DelParams.__doc__)

	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e	
	
	# if it got here, db should be open
	cur = database_conn.cursor()
	res = cur.execute("DELETE FROM tblModelParam WHERE model_id = %d AND param_date >= '%s' AND param_date <= '%s';"\
	%(model_id,from_date.isoformat(),to_date.isoformat()))
	database_conn.commit()
	cur.close()	
	
	return res


def PutSettings(model_id, setting_names, setting_values, setting_types):
	"""
	Save model settings into the database. Parameter values will be stored as strings
	(up to 200 characters).
	---
	Usage: result = PutSettings(model_id, setting_names, setting_values, setting_types)
	---
	model_id: numerical id of model in database
	setting_names: array_like of setting names to be stored as strings(20); no special
		characters, no spaces, nothing that might be interpreted!
	setting_values: array_like of values as same size, and in the same order, as names
	setting_types: array_like of python variable types stored as strings(10) as same size, and in
		the same order, as names; these can be used to convert from string back
	result: number parameters inserted into the database
	---
	ex: THIS SHOULD ONLY BE CALLED FROM A MODEL CLASS!!!
	JAH 20121103
	"""
	
	global database_open; global database_conn; global my_user
	
	# ensure input variables are correct
	if (type(model_id) is not int):
		raise TypeError("Variable(s) are wrong type: %s"%PutSettings.__doc__)
	if (len(setting_names) != len(setting_values)) or (len(setting_names) != len(setting_types)):
		raise ValueError("Number of setting names and values must match: %s"%PutSettings.__doc__)

	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e
			
	# delete any settings for this model first
	cur = database_conn.cursor()
	res = cur.execute("DELETE FROM tblModelSetting WHERE model_id = %d;"%model_id)
	database_conn.commit()
	
	# now add the settings
	# build the SQL first - we build a template with the id, and user
	SQLinsertT = "UNION ALL SELECT %s AS model_id, '%%s' as set_name, '%%s' as set_value, '%%s' as set_type ,'%s' as create_user, '%s' as create_date"%(model_id, my_user, dat.date.today())
	# now broadcast each name-value pair into the template, then catenate them all together to form the sql script
	SQLinserts = " ".join([SQLinsertT%(n,v,f) for n,v,f in zip(setting_names,setting_values,setting_types)])

	res = cur.execute("INSERT INTO tblModelSetting "+SQLinserts[10:-1]+";")
	database_conn.commit()
	cur.close()
	
	return res
	
	
def GetSettings(model_id):
	"""
	Get a model's stored settings.
	---
	Usage: names, values, types = GetSettings(model_id)
	---
	model_id: numerical id of model in database
	names: array of setting names
	values: array of values stored as strings as same size, and in the same order, as names
	types: array of python variable types stored as strings as same size, and in
		the same order, as names; these can be used to convert from string back
	---
	ex: 
	JAH 20121103
	"""

	global database_open; global database_conn

	# ensure input variables are correct
	if type(model_id) is not int:
		raise TypeError("Model id should be integer: %s"%GetSettings.__doc__)
	
	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e
			
	# if it got here, db should be open
	cur = database_conn.cursor()
	res = cur.execute("SELECT set_name, set_value, set_type FROM tblModelSetting WHERE model_id = %d;"%model_id)
	
	# maybe there can be no data
	if res == 0:
		cur.close()
		raise ValueError("No settings for model %d found!"%model_id)
		
	tmp = cur.fetchall(); cur.close()
	
	# extract the variables
	snams = np.array([x[0] for x in tmp],dtype=str)		# get setting names
	svals = np.array([x[1] for x in tmp],dtype=str)		# get setting values
	styps = np.array([x[2] for x in tmp],dtype=str)		# get setting types

	return snams,svals,styps	
	
	
def GetModels(model_name = None, target_index = None, training_days = None, trading_days = None, create_user=None):
	"""
	Return a tuple holding all the models in the QREDIS database that match certain
	criteria.
	---
	Usage: models = GetModels(model_name = None, target_index = None, training_days = None, trading_days = None, create_user=None)
	---
	model_name: filter on model class file name	
	target_index: filter on target index
	training / trading days: filter on training / trading days (integer)
	create_user: filter on who created the models
	models: tuple holding model details: id, name, target, training/trading/buffer days,
		source data, create user
	---
	ex: models = QD.GetModels(create_user = 'ahowe42')
	JAH 20120923
	"""

	global database_open; global database_conn

	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e
	
	# build the sql where clauses
	SQLwhere = ''
	if model_name is not None:
		SQLwhere += "model_name LIKE '%s' AND "%model_name
	if target_index is not None:
		SQLwhere += "target_index = '%s' AND "%target_index
	if create_user is not None:
		SQLwhere += "create_user = '%s' AND "%create_user
	if training_days is not None:
		SQLwhere += "training_days = %d AND "%training_days
	if trading_days is not None:
		SQLwhere += "trading_days = %d AND "%trading_days
	if len(SQLwhere) > 0:
		SQLwhere = " WHERE " + SQLwhere[:-5]
		
	# if it got here, db should be open
	cur = database_conn.cursor()
	res = cur.execute("SELECT model_id, model_name, target_index, training_days, trading_days, buffer_days, source_data, create_user FROM tblModel%s;"%SQLwhere)
	
	# maybe there can be no data
	if res == 0:
		cur.close()
		raise ValueError("No models found in QREDIS database matching that criteria!")
	
	# get the data from the cursor
	tmp = cur.fetchall(); cur.close()
	
	# just return results
	return tmp
# model functions

# general data functions
def CalRange():
	"""
	Return the first and last date in the QREDIS calendar.
	---
	Usage: dates = CalRange()
	---
	dates: array of 2 datetime.date values holding the first and last date in the QREDIS calendar
	---
	ex: minmax = QD.CalRange()
	JAH 20120929
	"""
	
	global database_open; global database_conn

	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e	
	
	# if it got here, db should be open
	cur = database_conn.cursor()
	res = cur.execute("SELECT min(cal_date) as mn, max(cal_date) as mx FROM tblCalendar;")
	
	# make sure we got something back
	if res == 0:
		cur.close()
		raise ValueError("Can't access the QREDIS Calendar!")
		
	tmp = cur.fetchone(); cur.close()	
	
	return np.array(tmp,dtype=dat.date)
	
	
def Calendar(from_date=dat.date(1000,1,1), to_date=dat.date(9999,12,31)):
	"""
	Return dates and holidays in a date range from the QREDIS calendar;
	if dates are not passed, all are returned.
	---
	Usage: dates, holidays = Calendar(from_date, to_date)
	---
	from_date* / to_date*: datetime.date first / last day required (inclusive)
	dates: array of datetime.date values holding the QREDIS calendar
	holidays: array of same size as dates holding string holiday designations
	---
	ex: dats,holi = QD.Calendar(dat.date(2012,1,1),dat.date(2012,1,31))
	JAH 20121001
	"""
	
	global database_open; global database_conn
	
	# check inputs
	if (type(from_date) is not dat.date) or (type(to_date) is not dat.date):
		raise TypeError("From and To dates must be datetime.date objects: %s"%Calendar.__doc__)

	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e	
	
	# if it got here, db should be open
	cur = database_conn.cursor()
	res = cur.execute("SELECT cal_date, holidays FROM tblCalendar WHERE cal_date >= '%s' AND cal_date <= '%s' ORDER BY cal_date;"%(from_date,to_date))
	
	# make sure we got something back
	if res == 0:
		cur.close()
		raise ValueError("Can't access the QREDIS Calendar!")

	# get the data from the cursor and prepare output
	tmp = cur.fetchall(); cur.close();
	dats = np.array([x[0] for x in tmp], dtype = dat.date)		# get the dates in one array ...
	holis = np.array([x[1] for x in tmp], dtype = str)			# ... the holiday strings in another
	
	return dats, holis
	
	
def GetTickers(filt_tick=None, filt_name=None, filt_holi=None, filt_curr=None, filt_type=None):
	"""
	Return data about the tickers in the QREDIS database.
	---
	Usage: tickers,tickdata = GetTickers(filt_tick, filt_name, filt_holi, filt_curr, filt_type)
	---
	filt_tick*: optional filter on ticker
	filt_name*: optional filter on index name
	filt_holi*: optional filter on holidays
	filt_curr*: optional filter on currency
	filt_type*: optional filter on index type
	tickers: array of selected tickers
	tickdata: tuple of tuples indicating, for each ticker: (ticker, name, currency code,
		holiday code, type, other_source,first CLOSE date, last CLOSE date)
	---
	ex: ticks,tickdata = QD.GetTickers(filt_tick="LIKE 'SP%'",filt_holi="NOT LIKE '%US%'")
	JAH 20121001
	"""
	
	global database_open; global database_conn

	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e	
	
	# build the extra where clauses
	SQLwhere = ''
	if filt_tick is not None:
		SQLwhere += "AND I.ticker %s "%filt_tick
	if filt_name is not None:
		SQLwhere += "AND I.name %s "%filt_name
	if filt_holi is not None:
		SQLwhere += "AND I.holidays %s "%filt_holi
	if filt_curr is not None:
		SQLwhere += "AND I.currency %s "%filt_curr
	if filt_type is not None:
		SQLwhere += "AND I.type %s "%filt_type
	
	# if it got here, db should be open
	cur = database_conn.cursor()
	res = cur.execute("SELECT I.ticker, name, currency, holidays, type, other_source, D.mn, D.mx FROM tblIndex as I INNER JOIN "\
	"(SELECT ticker, min(price_date) as mn, max(price_date) as mx FROM tblIndexDaily WHERE close > 0 GROUP BY ticker) AS D On "\
	"I.ticker = D.ticker WHERE I.ticker <> '_CHEAT' %s ORDER BY I.ticker;"%SQLwhere)
	
	# make sure we got something back
	if res == 0:
		cur.close()
		if SQLwhere == '':
			raise ValueError("Can't get tickers data!")
		else:
			raise ValueError("Can't find tickers matching: %s"%SQLwhere[4:])
		
	tickdata = cur.fetchall(); cur.close()
	
	# parse the list of tickers
	tickers = np.array([x[0] for x in tickdata],dtype=str)

	return tickers, tickdata
# general data functions

# QREDIS data inserting/editing functions
def LoadCSVData(filename, date_format='%d-%b-%Y'):
	"""
	Load CSV price data in the format of TICKER,Date,open,high,low,close,volume.
	---
	Usage: records = LoadCSVData(filename, date_format)
	---
	filename: full path & filename of the file to parse into the database
	date_format*: dat.date string format of date in file; default is '%d-%b-%Y'
	records_inserted: number records inserted into the database
	---
	ex: res = QD.LoadCSVData(QREDIS_dat+'/EODData/FOREX_20140108.csv','%d-%b-%Y')
	JAH 20130404
	"""
	
	global database_open; global database_conn; global my_user

	# ensure input variables are correct
	try:
		jnk = filename.upper()
		jnk = date_format.upper()
	except AttributeError:
		raise TypeError("Input file name and date format must be strings: %s"%LoadCSVData.__doc__)

	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e
	
	# if it got here, db should be open
	# JAH changed from using np.loadtxt 20140108
	#try:
	#	inp = (np.loadtxt(filename, dtype='|S11', delimiter=',',skiprows=1)).astype(str)
	#except IOError:
	#	raise IOError("%s does not exist, or can't be accessed!"%filename)
	# parse columns
	#n = inp.shape[0]
	#dats = np.asarray([dat.datetime.strptime(jnk,date_format) for jnk in inp[:,1]],dtype=dat.date)
	#prics = np.asarray(inp[:,2:],dtype=float)
	
	# read in file
	try:
		inp = open(filename).read()
	except IOError:
		raise IOError("%s does not exist, or can't be accessed!"%filename)
	# separate the bigass string into a) a row for each row in the data, then
	# b) a cell for each csv value
	inp = [row.split(',') for row in inp.splitlines()[1:]]
	n = len(inp)
	# separate the dates, prices, and tickers into their own arrays
	dats = np.asarray([dat.datetime.strptime(row[1],date_format).date().isoformat() for row in inp],dtype=str)
	prics = np.asarray([row[2:] for row in inp],dtype=float)
	inp = np.asarray([row[0] for row in inp],dtype=str)

	# map tickers as specified
	# don't load NDY, SPX, XBEL JAH 20130404 (dups of NDX, SP500, BEL20)
	# JAH also don't load FTMC (dup of FTMS), NKY (dup of NI225) 20130716
	# JAH 20140110 don't load XU100
	keep = ~((inp == 'SPX') | (inp == 'XBEL') | (inp == 'FTMC') | (inp == 'NKY') | (inp == 'XU100'))	
	# JAH rename NDY to NDX since it is a dup and sometimes there is NDY but not NDX 20130802
	inp[inp == 'NDY']='NDX'
	
	# JAH 20160205 - check for any new tickers first
	missing = ''
	# create the select
	uniticks = np.unique(inp[keep]).tolist()
	SQL = 'SELECT ticker FROM tblIndex WHERE ticker IN ('+('%s'%uniticks)[1:-1]+');'
	# execute
	cur = database_conn.cursor()
	res = cur.execute(SQL)
	# check if anything returned at all
	if res == 0:
		cur.close()
		t = ('%s'%(inp[keep].tolist()))[2:-2].replace("'",'')	# don't need the quotes either
		missing += '\nLooks like there are %d new tickers: %s'%(len(inp[keep]),t)
	else:
		res = np.array([x[0] for x in cur.fetchall()],dtype=str)
		cur.close()
		# now check that number records is as expected
		if len(res) != len(uniticks):
			tofind = len(uniticks) - len(res)
			missing += '\nLooks like there are %d new tickers: '%(tofind)
			for t in uniticks:
				if sum(res == t) == 0:
					# not found, so add to missing message and decrement counter
					missing = missing + t + ', '
					tofind -= 1
					# see if we've accounted for all new tickers and can break out
					if tofind == 0:
						break

	# JAH 20160205 - now check for any new dates
	# create the select
	unidats = np.unique(dats[keep]).tolist()
	SQL = 'SELECT cal_date FROM tblCalendar WHERE cal_date IN ('+('%s'%unidats)[1:-1]+');'
	# execute
	cur = database_conn.cursor()
	res = cur.execute(SQL)
	# check if anything returned at all
	if res == 0:
		cur.close()
		t = ('%s'%unidats)[2:-2].replace("'",'')	# don't need the quotes either
		missing += '\nLooks like there are %d new dates: %s'%(len(unidats),t)
	else:	
		res = np.array([x[0] for x in cur.fetchall()],dtype=str)
		cur.close()
		# now check that number records is as expected
		if len(res) != len(unidats):
			tofind = len(unidats) - len(ticks)
			missing += '\nLooks like there are %d new dates: '%(tofind)
			for t in unidats:
				if sum(res == t) == 0:
					# not found, so add to missing message and decrement counter
					missing = missing + t + ', '
					tofind -= 1
					# see if we've accounted for all new dates and can break out
					if tofind == 0:
						break
					
	# JAH 20160205 - if anything was missing, display the message and exit
	if missing != '':
		raise DataError(missing[1:]+"\nPlease add what's missing and then run LoadCSVData('%s', '%s')"%(filename, date_format))
	
	# create SQL insert commands	
	SQLinsertT = "UNION ALL SELECT '%s' AS ticker, '%s' AS price_date, %s AS open, %s AS high, %s AS low, %s AS close, %s AS volume, NULL as notes "
	SQLinserts = " ".join([SQLinsertT%(t,d,p[0],p[1],p[2],p[3],p[4]) for t,d,p in \
	zip(inp[keep],dats[keep],prics[keep,:])])
	
	# now execute it all
	cur = database_conn.cursor()	
	res = cur.execute("INSERT IGNORE INTO tblIndexDaily "+SQLinserts[10:-1]+";")
	database_conn.commit()
	cur.close()
	print('Records:\nloaded = %d\nexcluded = %d\ninserted = %d'%(n,n-np.sum(keep),res))
	
	# finally, close the db; next call to the db will reopen it, and new data will be available
	CloseDB()
	
	return res
	
def LoadXU100Data(from_date):
	"""
	Download the XU100 prices from the ISE website and import them into the QREDIS database
	starting from the from_date.
	---
	Usage: records = LoadXU100Data(from_date)
	---
	from_date: dat.date() date, all data available after this date will be imported
	records: number records inserted into the database
	---
	ex: res = QD.LoadXU100Data(dat.date(2014,1,1))
	JAH 20140107, adopted from freelancer.com contract work of ghazalpasha
	"""
	
	global database_open; global database_conn; global my_user
	
	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e	

	# if it got here, db should be open
	# ensure input variables are correct
	if type(from_date) is not dat.date:
		raise TypeError("from_date must be dat.date: %s"%LoadXU100Data.__doc__)
	
	# DOWNLOAD THE FILE
	try:
		origurl = 'http://borsaistanbul.com/veriler/verileralt/hisse-senetleri-piyasasi-verileri/endeks-verileri'

		# get download page
		http = httplib2.Http()
		url = origurl
		headers = {
			'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
			'Accept-Language': 'en-US,en;q=0.8',
			'Accept-Encoding': 'gzip,deflate,sdch',#'deflate,sdch', JAH changed 20141104
			'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
			'User-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11'}
		response, content = http.request(url, 'GET', headers=headers)

		# handle redirects
		while 'http-equiv="refresh"' in str(content):
			url = re.search(r'url=([^"]*)"', str(content)).group(1)
			response, content = http.request(url, 'GET', headers=headers)

		# add headers to download file
		headers['Cookie'] = response['set-cookie']
		headers['Referer'] = url
		headers['origin'] = 'http://borsaistanbul.com'
		headers['Host'] = 'borsaistanbul.com'
		headers['Content-type'] = 'application/x-www-form-urlencoded'

		# get form data and simulate click by adding items to form data
		form = dict(re.findall(r'id="([^"]*)" value="([^"]*)"', str(content)))
		form["ctl00$TextContent$C001$lbtnFiyatEndeksleri.x"] = "1"
		form["ctl00$TextContent$C001$lbtnFiyatEndeksleri.y"] = "1"

		# download zip file
		url = origurl
		response, zipdata = http.request(url, 'POST', headers=headers, body=urllib.parse.urlencode(form))

		# unzip and return csv
		zipfile = ZipFile(BytesIO(zipdata))
		csv = zipfile.read(zipfile.infolist()[0])
	except:
		print("Unexpected error while downloading data:", sys.exc_info()[0])
		return 0
    
    # PREPARE THE DATA
	try:
		# get data portion of csv
		data = [[item.strip() for item in line.decode().split(";")] for line in csv.splitlines()[4:]]

		# append dummy at the end
		data.append(["0","0","0","0","0","0","0","0"])
		processed_data = []

		# go row by row in csv
		# when a new date is seen, append the final info of last date to processed data
		last_row_date = ""
		last_day_final_close = 0
		for row in data:
			(ticker, date, session, low, high, close, usd, euro) = row
			if date != last_row_date:
				# write data for last row date
				if last_row_date != "":
					# date, open, high, low, close, close_usd, close_euro
					processed_data.append((dat.datetime.strptime(last_row_date,"%d/%m/%y").date(),
					last_day_final_close, max_high, min_low, final_close, final_usd, final_euro))
					last_day_final_close = final_close
    
				# start new low/high
				min_low = float(low)
				max_high = float(high)
			else:
				# take min for low, max for high
				min_low = min(min_low, float(low))
				max_high = max(max_high, float(high))
    
			final_close = float(close)
			final_usd = float(usd)
			final_euro = float(euro)

			last_row_date = date
	except:
		print("Unexpected error while processing data:", sys.exc_info()[0])
		return 0
        
	# CREATE AND EXECUTE THE SQL INSERTS
	# sql format string JAH changed from original submission to make more efficient inserts 20140108
	sql = "UNION ALL SELECT 'XU100_TL' as Ticker, '%s' as price_date, %0.2f as open, %0.2f as high, %0.2f as low, %0.2f as close, 0 as volume, NULL as notes " \
	"UNION ALL SELECT 'XU100_US' as Ticker, '%s' as price_date, 0 as open, 0 as high, 0 as low, %0.2f as close, 0 as volume, NULL as notes " \
	"UNION ALL SELECT 'XU100_EU' as Ticker, '%s' as price_date, 0 as open, 0 as high, 0 as low, %0.2f as close, 0 as volume, NULL as notes"

	# return sql statement for all rows in processed_data with date >= in_date
	SQLs = [sql % (date.isoformat(), tl_open, high, low, tl_close, date.isoformat(), usd_close, date.isoformat(), euro_close)
			for (date, tl_open, high, low, tl_close, usd_close, euro_close) in processed_data
			if not date < from_date]
		
	# now run the inserts JAH changed from original submission to make more efficient inserts 20140108
	cur = database_conn.cursor()
	res = cur.execute("INSERT IGNORE INTO tblIndexDaily " +(" ".join(SQLs))[10:]+";")
	
	database_conn.commit()
	cur.close()
	print('Records inserted = %d'%res)
	
	# finally, close the db; next call to the db will reopen it, and new data will be available
	CloseDB()
	
	return res

def LoadEODData(price_date, types=['INDEX','FOREX']):
	"""
	Download prices from the EOD website and import them into the QREDIS database for a
	a specified date.
	---
	Usage: records = LoadEODData(price_date, types)
	---
	price_date: dat.date() date, download all INDEX and FOREX data for this date
	types*: list of strings of either 'INDEX','FOREX', or both*
	records: number records inserted into the database
	---
	ex: res = QD.LoadEODData(dat.date(2014,1,1),['INDEX'])
	JAH 20140110, adopted from freelancer.com contract work of ghazalpasha
	"""
	
	global database_open; global database_conn; global my_user
	
	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e	

	# if it got here, db should be open
	# ensure input variables are correct
	try:
		jnk = ('FOREX' in types) or ('INDEX' in types)
	except Exception as e:
		raise e("types must be a list of strings: %s"%LoadEODData.__doc__)
	if (not(jnk)) or (type(types) is not list):
		raise ValueError("types must be a list of strings including 'INDEX' and/or 'FOREX': %s"%LoadEODData.__doc__)
	if type(price_date) is not dat.date:
		raise TypeError("price_date must be dat.date: %s"%LoadEODData.__doc__)

	# ensure the EODData subfolder exists
	if not(os.path.exists(__builtins__['QREDIS_dat']+'/EODData')):
		os.mkdir(__builtins__['QREDIS_dat']+'/EODData')

	# connect to website and login and all that junk
	try:
		download_page_url = '/download.aspx'
		download_url = '/data/filedownload.aspx?e=%s&sd=%s&ed=%s&d=4&o=d&ea=1&p=0&k=%s'
		host = 'www.eoddata.com'

		headers = {
			'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
			'Accept-Language': 'en-US,en;q=0.8',
			'Accept-Encoding': 'deflate,sdch',
			'Cache-Control': 'max-age=0',
			'Connection': 'keep-alive',
			'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
			'User-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/31.0.1650.63 Safari/537.36'}

		# get login page
		conn = http.client.HTTPConnection(host,80)
		conn.request("GET","/","",headers)
		response = conn.getresponse()
		content = response.read()

		# update headers
		headers['Cookie'] = response.info().get("Set-Cookie")
		headers['Content-Type'] = 'application/x-www-form-urlencoded'

		# get form data for login POST
		form = dict(re.findall(r'id="([^"]*)" value="([^"]*)"', content.decode()))
		form["ctl00$cph1$lg1$txtEmail"] = 'ahowe42'
		form["ctl00$cph1$lg1$txtPassword"] = 'QREDIS2012'
		form["ctl00$cph1$lg1$btnLogin"] = "Login"
		form["ctl00$Menu1$s1$txtSearch"] = ""

		# login POST request
		conn = http.client.HTTPConnection(host,80)
		conn.request("POST","/",urllib.parse.urlencode(form),headers)
		response = conn.getresponse()
		content = response.read()

		# update headers
		headers['Cookie'] = headers['Cookie'] + "; " + response.info().get("Set-Cookie")

		# get download page
		conn = http.client.HTTPConnection(host,80)
		conn.request("GET",download_page_url,"",headers)
		response = conn.getresponse()
		content = response.read()

		# extract key from download page
		key = re.search(r'&k=([^"]*)"', content.decode()).group(1)

	except:
		print("Unexpected error while connecting to website:", sys.exc_info()[1])
		return 0
		
	# connected to the website so do the downloads and imports
	res = 0
	file_name = '%s/EODData/%s_%s.csv'%(__builtins__['QREDIS_dat'],'%s',price_date.strftime('%Y%m%d'))
	
	for cnt in range(len(types)):
		success = False
		try:
			fn = file_name%(types[cnt])
			conn = http.client.HTTPConnection(host,80)
			conn.request("GET",download_url % (types[cnt], price_date.strftime('%Y%m%d'), \
				price_date.strftime('%Y%m%d'), key),"",headers)
			response = conn.getresponse()
			content = response.read()
		except:
			print("Unexpected error while downloading %s data:"%(types[cnt]), sys.exc_info()[1])
			return 0
		try:
			# now save it if anything came
			if len(content) > 0:
				# ensure file doesn't already exist
				if os.path.exists(fn):
					print('%s already exists! skipping ...'%fn)
				else:
					with open(fn, "wb") as out_file:
						out_file.write(content)
					success = True
			else:
				# no data
				print('No %s data found for %s'%(types[cnt],price_date))
		except IOError:
			raise IOError("Can't save %s data to %s!"%(types[cnt],fn))
		if success:
			print('\n%s data downloaded to %s!'%(types[cnt],fn))
			# JAH 20160205 added the try-except block
			try:
				res += LoadCSVData(fn, date_format='%d-%b-%Y')
			except DataError as e:
				# simply print error messag and continue
				print(e.args[0])				

	return res

def PutTickerPrice(ticker, price_date, source, price_open, price_high, price_low, price_close, volume):
	"""
	Insert a ticker price from a non-automatic-download source (i.e, bloomberg, google, etc)..
	---
	Usage: records = PutTickerPrice(ticker, price_date, source, price_open, price_high, price_low, price_close)
	---
	ticker: string ticker to insert
	price_date: dat.date() price date to insert
	source: name of source, i.e. Bloomberg or Google; will have user name and date added, then truncated to 50
	price_open: opening price
	price_high: high price
	price_low: low price
	price_close: closing price
	volume: trading volume
	records: number records inserted into the database; should be 1
	---
	ex: res = QD.PutTickerPrice('BDI',dat.date(2014,1,13),'Bloomberg',2145,2145,2145,2145,0)
	JAH 20140101
	"""

	global database_open; global database_conn; global my_user
	
	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e	

	# if it got here, db should be open
	# check that price inputs are numerical by trying to add them
	try:
		jnk = price_open + price_high + price_low + price_close + volume
	except TypeError:
		raise TypeError("Open, High, Low, Close, and Volume must all be numeric: %s"%PutTickerPrice.__doc__)
	# simultaneously build edit note and check string inputs
	try:
		ticker = ticker.upper()
		note = "%s %s %s"%(my_user,dat.date.today(),source)[:50]
	except AttributeError:
		raise TypeError("Ticker must be a string: %s"%PutTickerPrice.__doc__)

	# ensure input variables are correct
	if type(price_date) is not dat.date:
		raise TypeError("price_date must be dat.date: %s"%PutTickerPrice.__doc__)
		
	# ensure a) ticker is in tblIndex ...
	cur = database_conn.cursor()
	res = cur.execute("SELECT ticker from tblIndex where ticker = '%s';"%ticker)
	if res == 0:
		cur.close()
		raise ValueError("Can't find ticker %s!"%ticker)
	
	# ... and b) price_date is in tblCalendar ...
	res = cur.execute("SELECT cal_date from tblCalendar where cal_date = '%s';"%price_date.isoformat())
	if res == 0:
		cur.close()
		raise ValueError("Can't find %s in QREDIS calendar!"%price_date)
		
	# ... and c) that there isn't a record in tblIndexDaily already
	res = cur.execute("SELECT ticker FROM tblIndexDaily WHERE ticker = '%s' AND price_date = '%s'"%(ticker,price_date.isoformat()))
	if res != 0:
		cur.close()
		raise ValueError("Price record already exists on %s for %s!"%(price_date,ticker))	
	
	# execute the insert into the log table ...
	res = cur.execute("INSERT INTO tblIndexDaily_log values('%s','%s',%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,'%s','%s','ADD','%s');"\
	%(ticker, price_date.isoformat(), price_open, price_high, price_low, price_close,volume,dat.date.today().isoformat(),my_user,source[:50]))
	# ... then insert it
	res = cur.execute("INSERT INTO tblIndexDaily values('%s','%s',%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,'%s');"\
	%(ticker, price_date.isoformat(), price_open, price_high, price_low, price_close,volume,note))
	database_conn.commit()
	cur.close()
	
	# make sure records were inserted
	if res == 0:
		raise ValueError("Unable to insert prices on %s for %s!"%(price_date,ticker))
	
	return res
	
def DelTickerPrice(ticker, price_date, note):
	"""
	Delete a ticker price from the QREDIS database.
	---
	Usage: records = DelTickerPrice(ticker, price_date, note)
	---
	ticker: string ticker to delete
	price_date: dat.date() price date to delete
	note: string reason for delete; will have user name added and truncated to 50 chars
	records: number records deleted from the database; should be 1
	---
	ex: res = QD.DelTickerPrice('BDI',dat.date(2014,1,13),'just an example')
	JAH 20140101
	"""

	global database_open; global database_conn; global my_user
	
	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e	

	# duck type ticker, maybe it's np.string, by trying to uppercase it
	# this won't work if it's not a string-type variable JAH 20130320
	try:
		ticker = ticker.upper()
	except AttributeError:
		raise TypeError("Ticker must be a string: %s"%DelTickerPrice.__doc__)
		
	# if it got here, db should be open
	# ensure input variables are correct
	if (type(price_date) is not dat.date):
		raise TypeError("price_date must be dat.date: %s"%DelTickerPrice.__doc__)

	# ensure there is a record in tblIndexDaily for this ticker & date
	cur = database_conn.cursor()
	res = cur.execute("SELECT ticker from tblIndexDaily where ticker = '%s' and price_date = '%s';"\
	%(ticker,price_date.isoformat()))
	if res == 0:
		cur.close()
		raise ValueError("Can't find prices for ticker %s and price_date %s!"%(ticker,price_date))
		
	# execute the insert into the log table ...
	res = cur.execute("INSERT INTO tblIndexDaily_log SELECT ticker,price_date,open,high,low,close,volume,'%s','%s','DEL','%s' FROM tblIndexDaily WHERE ticker = '%s' AND price_date = '%s';"\
	%(dat.date.today().isoformat(),my_user,note[:50],ticker, price_date.isoformat()))
	# ... then delete it
	res = cur.execute("DELETE FROM tblIndexDaily where ticker = '%s' and price_date = '%s';"\
	%(ticker, price_date.isoformat()))
	database_conn.commit()
	cur.close()
	
	# make sure records were deleted
	if res == 0:
		raise ValueError("No prices on %s for %s deleted!"%(price_date,ticker))
	
	return res

def EdtTickerPrice(ticker, price_date, source, price_open=None, price_high=None, price_low=None, price_close=None, volume=None):
	"""
	Edit a ticker price from a non-automatic-download source (i.e, bloomberg, google, etc),
	or to correct an obvious scaling error.
	---
	Usage: records = EdtTickerPrice(ticker, price_date, source, price_open, price_high, price_low, price_close)
	---
	ticker: string ticker to insert
	price_date: dat.date() price date to insert
	source: string ame of source, i.e. Bloomberg or Google or Logic, Spike, etc; will have
		user name and date added, then truncated to 50
	price_open*: numeric opening price
	price_high*: numeric high price
	price_low*: numeric low price
	price_close*: numeric closing price
	volume*: numeric trading volume
	records: integer number records edited in the database; should be 1
	---
	ex: res = QD.EdtTickerPrice('BDI',dat.date(2014,1,13),'Bloomberg',price_close=2142)
	JAH 20140101
	"""

	global database_open; global database_conn; global my_user
	
	# make sure database is open, if not, try to open
	if database_open == False:
		try:
			OpenDB()
		except Exception as e:
			print("OpenDB error occurred - can't open QREDIS Database!")
			raise e	

	# if it got here, db should be open
	# check that price inputs are numerical by trying to add them and build the edit string
	try:
		jnk = 0
		edt = ''
		if price_open is not None:
			jnk += price_open
			edt += 'O'
		if price_high is not None:
			jnk += price_high
			edt += 'H'
		if price_low is not None:
			jnk += price_low
			edt += 'L'
		if price_close is not None:
			jnk += price_close
			edt += 'C'
		if volume is not None:
			jnk += volume
			edt += 'V'
	except TypeError:
		raise TypeError("Open, High, Low, Close, and Volume must all be numeric if provided: %s"%EdtTickerPrice.__doc__)
	# simultaneously build edit note and check string inputs
	try:
		ticker = ticker.upper()
		note = "%s %s edit %s %s"%(my_user,dat.date.today(),edt,source)[:50]
	except AttributeError:
		raise TypeError("Ticker and note must be a string: %s"%EdtTickerPrice.__doc__)

	# ensure input variables are correct
	if type(price_date) is not dat.date:
		raise TypeError("price_date must be dat.date: %s"%EdtTickerPrice.__doc__)
	if len(edt) == 0:
		raise ValueError("Must enter at least 1 of Open, High, Low, Close, Volume: %s"%EdtTickerPrice.__doc__)

	# ensure there is already a record for this ticker and price_date in tblIndexDaily
	cur = database_conn.cursor()
	res = cur.execute("SELECT ticker FROM tblIndexDaily WHERE ticker = '%s' AND price_date = '%s'"\
	%(ticker,price_date.isoformat()))
	if res == 0:
		cur.close()
		raise ValueError("No price record on %s for %s, use QD.PutTickerPrice!"%(price_date,ticker))	
	
	# execute the insert into the log table ...
	res = cur.execute("INSERT INTO tblIndexDaily_log SELECT ticker,price_date,open,high,low,close,volume,'%s','%s','EDT','%s' FROM tblIndexDaily WHERE ticker = '%s' AND price_date = '%s';"\
	%(dat.date.today().isoformat(),my_user,(edt+' '+note)[:50],ticker, price_date.isoformat()))
	# ... then do the edits
	if price_open is not None:
		res = cur.execute("UPDATE tblIndexDaily set open = %0.4f where ticker = '%s' and price_date = '%s';"\
		%(price_open, ticker, price_date.isoformat()))
	if price_high is not None:
		res = cur.execute("UPDATE tblIndexDaily set high = %0.4f where ticker = '%s' and price_date = '%s';"\
		%(price_high, ticker, price_date.isoformat()))
	if price_low is not None:
		res = cur.execute("UPDATE tblIndexDaily set low = %0.4f where ticker = '%s' and price_date = '%s';"\
		%(price_low, ticker, price_date.isoformat()))
	if price_close is not None:
		res = cur.execute("UPDATE tblIndexDaily set close = %0.4f where ticker = '%s' and price_date = '%s';"\
		%(price_close, ticker, price_date.isoformat()))
	if volume is not None:
		res = cur.execute("UPDATE tblIndexDaily set volume = %0.4f where ticker = '%s' and price_date = '%s';"\
		%(volume, ticker, price_date.isoformat()))

	res = cur.execute("UPDATE tblIndexDaily set notes = '%s' where ticker = '%s' and price_date = '%s';"\
	%(note, ticker, price_date.isoformat()))
	database_conn.commit()
	cur.close()
	
	# make sure records were edited
	if res == 0:
		raise ValueError("Unable to edit prices on %s for %s!"%(price_date,ticker))
	
	return res
# QREDIS data inserting/editing functions
