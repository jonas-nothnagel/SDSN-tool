import configparser
import logging

def getconfig(configfile_path:str):
    """
    configfile_path: file path of .cfg file
    """

    config = configparser.ConfigParser()

    try:
        config.read_file(open(configfile_path))
        return config
    except:
        logging.warning("config file not found")