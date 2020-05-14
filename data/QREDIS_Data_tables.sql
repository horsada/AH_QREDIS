-- MySQL dump 10.13  Distrib 5.7.13, for Linux (x86_64)
--
-- Host: localhost    Database: QREDIS_Data
-- ------------------------------------------------------
-- Server version	5.7.13-0ubuntu0.16.04.2

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `tblCalendar`
--

DROP TABLE IF EXISTS `tblCalendar`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `tblCalendar` (
  `cal_date` date NOT NULL,
  `holidays` varchar(250) DEFAULT NULL,
  PRIMARY KEY (`cal_date`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `tblIndex`
--

DROP TABLE IF EXISTS `tblIndex`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `tblIndex` (
  `ticker` varchar(10) NOT NULL,
  `name` varchar(100) NOT NULL,
  `currency` char(3) NOT NULL,
  `holidays` varchar(20) NOT NULL,
  `type` varchar(50) DEFAULT NULL,
  `other_source` varchar(25) DEFAULT NULL,
  PRIMARY KEY (`ticker`),
  KEY `Type` (`type`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `tblIndexDaily`
--

DROP TABLE IF EXISTS `tblIndexDaily`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `tblIndexDaily` (
  `ticker` varchar(10) NOT NULL DEFAULT '',
  `price_date` date NOT NULL,
  `open` double(15,4) DEFAULT NULL,
  `high` double(15,4) DEFAULT NULL,
  `low` double(15,4) DEFAULT NULL,
  `close` double(15,4) NOT NULL,
  `volume` bigint(10) unsigned DEFAULT NULL,
  `notes` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`ticker`,`price_date`),
  KEY `id_Date` (`price_date`),
  CONSTRAINT `fk_ticker` FOREIGN KEY (`ticker`) REFERENCES `tblIndex` (`ticker`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `tblIndexDaily_ibfk_1` FOREIGN KEY (`price_date`) REFERENCES `tblCalendar` (`cal_date`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `tblIndexDaily_log`
--

DROP TABLE IF EXISTS `tblIndexDaily_log`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `tblIndexDaily_log` (
  `ticker` varchar(10) NOT NULL DEFAULT '',
  `price_date` date NOT NULL,
  `open` double(15,4) DEFAULT NULL,
  `high` double(15,4) DEFAULT NULL,
  `low` double(15,4) DEFAULT NULL,
  `close` double(15,4) NOT NULL,
  `volume` bigint(10) unsigned DEFAULT NULL,
  `log_date` date NOT NULL,
  `log_user` varchar(50) DEFAULT NULL,
  `log_type` char(3) NOT NULL DEFAULT '',
  `log_notes` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `tblModel`
--

DROP TABLE IF EXISTS `tblModel`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `tblModel` (
  `model_id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `model_name` varchar(50) NOT NULL,
  `model_descrip` varchar(500) DEFAULT NULL,
  `target_index` varchar(10) NOT NULL,
  `training_days` int(11) unsigned NOT NULL,
  `trading_days` int(11) unsigned NOT NULL,
  `buffer_days` int(10) unsigned DEFAULT NULL,
  `source_data` varchar(1000) DEFAULT NULL,
  `create_user` varchar(50) NOT NULL,
  PRIMARY KEY (`model_id`)
) ENGINE=InnoDB AUTO_INCREMENT=32 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `tblModelParam`
--

DROP TABLE IF EXISTS `tblModelParam`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `tblModelParam` (
  `model_id` int(11) unsigned NOT NULL,
  `param_date` date NOT NULL,
  `param_name` varchar(20) NOT NULL,
  `param_value` varchar(200) NOT NULL,
  `param_type` varchar(10) NOT NULL,
  `run_date` date NOT NULL,
  `run_user` varchar(50) NOT NULL,
  PRIMARY KEY (`model_id`,`param_date`,`param_name`),
  KEY `fk_param_date` (`param_date`),
  CONSTRAINT `fk_modelid` FOREIGN KEY (`model_id`) REFERENCES `tblModel` (`model_id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_param_date` FOREIGN KEY (`param_date`) REFERENCES `tblCalendar` (`cal_date`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `tblModelSetting`
--

DROP TABLE IF EXISTS `tblModelSetting`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `tblModelSetting` (
  `model_id` int(10) unsigned NOT NULL,
  `set_name` varchar(20) NOT NULL,
  `set_value` varchar(200) NOT NULL,
  `set_type` varchar(10) NOT NULL,
  `create_user` varchar(50) NOT NULL,
  `create_date` date NOT NULL,
  PRIMARY KEY (`model_id`,`set_name`),
  CONSTRAINT `tblModelSetting_ibfk_1` FOREIGN KEY (`model_id`) REFERENCES `tblModel` (`model_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `tblModelSignal`
--

DROP TABLE IF EXISTS `tblModelSignal`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `tblModelSignal` (
  `model_id` int(11) unsigned NOT NULL,
  `signal_date` date NOT NULL,
  `signal_value` tinyint(4) NOT NULL,
  `signalgen_flag` char(2) NOT NULL DEFAULT 'BT',
  `run_date` date NOT NULL,
  `run_user` varchar(50) NOT NULL,
  PRIMARY KEY (`model_id`,`signal_date`),
  KEY `fk_signal_date` (`signal_date`),
  CONSTRAINT `fk_modelid1` FOREIGN KEY (`model_id`) REFERENCES `tblModel` (`model_id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_signal_date` FOREIGN KEY (`signal_date`) REFERENCES `tblCalendar` (`cal_date`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;