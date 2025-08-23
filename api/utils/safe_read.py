from api.utils.quotas import reserve_or_reject, credit_back
import pandas as pd


def safe_read_csv(file_path, ip_address, requested_rows):
    """
    read a csv file, if failes refund rows
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        credit_back(ip_address, requested_rows)
        return None
