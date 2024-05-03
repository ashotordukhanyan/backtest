import logging
from utils.qpthack import qconnection
from market_data.extractalpha.datasets import EATM,EACAM

if __name__ == '__main__':
    logging.basicConfig(filename=None,level=logging.INFO,format='%(levelname)s %(asctime)s %(message)s')
    LOAD_TM = False
    LOAD_CAM = True
    with qconnection.QConnection('localhost', 12345, pandas=True) as q:
        KDB_ROOT = "c:/KDB_MARKET_DATA2/"
        if LOAD_TM:
            logging.info('Loading tactial model to KDB')
            TM_FILE = r'C:\Users\orduk\OneDrive\Documents\ExtractAlpha\TM1_History_2000_202312.zip'
            tm = EATM()
            tm.load_df_to_kdb(TM_FILE,q,KDB_ROOT)
        if LOAD_CAM:
            logging.info('Loading cross asset model to KDB')
            CAM_FILE = r'C:\Users\orduk\OneDrive\Documents\ExtractAlpha\CAM1_History_2005_202312.zip'
            cam = EACAM()
            cam.load_df_to_kdb(CAM_FILE, q, KDB_ROOT)
