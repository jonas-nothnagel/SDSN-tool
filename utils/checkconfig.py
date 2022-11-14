import configparser
import logging

def getconfig(configFilePath):

    config = configparser.ConfigParser()

    try:
        config.read_file(open(configFilePath))
        return config
    except:
        logging.warning("config file not found")