# QREDIS
## History
Between 1999 and 2003, while working at Golden Gate Financial Group, I was responsible for researching and developing market timing trading models. I did this work in collaboration with Mr. Tom Write, president of Applied Market Analytics. Tom had personally developed a very cool modeling heuristic platform for researching and testing trading models, based on the *APL* programming language. It was called the **QCL** - *Q*uant *C*ommand *L*anguage.

While at Golden Gate, I developed many trading models in the QCL, and extended it's functionality in several areas, including:

- calculating / reporting model metrics, and adaptive model scoring
- transforming financial time-series data to analyzable signals
- automated time-adaptive searching through modeling parameter space

For years after working there, I wanted to build my own similar system. This is it.

**QREDIS** (pronounced *kredis*) is a modeling heuristic platform + database which I built in python with a MySQL backend between 2012 and 2014. It includes the following capabilities:

- calendar and data management
- rapid development
- mathematical / statistical tools and adaptive walk-forward modeling
- back-testing
- easy extensibility with new methods
- model / target charting and reporting
- real-time signal generation
- stress and scenario testing
- model and signals version control

The name is an acronym for: *Q*uantitative
*RE*search
and
*D*evelopment
*I*nformation
*S*ystem

## Organization
QREDIS is extensible in that the code is arranged in a modular architecture. As well as adding functions to the existing modules, new ones can be easily integrated. Current modules in the *src* directory are:

- QREDIS_Basic - basic generic functions
- QREDIS_Data - functions for reading / writing data
- QREDIS_Model - functions for running / testing / evaluating trading models
- QREDIS_GA - *genetic algorithm* optimization functions
- QREDIS_INFCOMP - functions related to computing *information-theoretic* model selection criteria

Models in QREDIS are defined by extending a base model class [*QMod_Template*](/models/QMod_Template.py) and overloading specified methods. For demonstrative purposes, I built a useless [example model](/models/QMod_Example.py). While working in QREDIS, I also developed several real models:

- [QMod_PersRever_Select](/models/QMod_PersRever_Select.py) & [QMod_PersRever_Select](/models/QMod_PersReverRet_Select.py): Evaluate a training period of a specific index (presumably the target index) and count the number of days in each sign (up1, up2, dn1, up1, dn1, dn2, dn3, ...) based on of occurrences of each in-direction day, and the relative strength of the tendency to persist (1 -> 2 or -2 -> -3) or reverse (1 -> -1 or -2 -> 1), identify a subset of these tendencies to trade during the trading period.
- [QMod_RegressGASubPred](/models/QMod_RegressGASubPred.py): This model trades the target index by selecting, using the GA & information criteria, a subset regression model to predict the daily returns. X is composed of lags of daily returns of the source data.

## Data
Trading models are big consumers of data, and the *QREDIS_Data* MySQL database is a fundamental component of the system. The database stored international indices & their daily values, daily currency exchange rates, holiday calendar designations, models, model parameters, and model signals. QREDIS is designed to be able to work with indices from many different national stock markets, so the ability to handle disparate trading calendars was built in from the beginning. A lot of research went into identifying trading holidays. I have documented at least some of the source websites I used [here](/data/international_stockmarket_tradingcalendar_sources.txt). The database table definitions are defined in [this file](/data/QREDIS_Data_tables.sql).

## Code
I wrote much of QREDIS while I was relatively new to Python. I'm sure there is much I could have done more efficiently, and especially using [pandas](https://pandas.pydata.org/). *Caveat emptor*!





