from api.utils.quotas import credit_back
import pandas as pd


def safe_read_csv(file_path, quota_subject, requested_rows):
    """
    Read a CSV file; if it fails, refund rows to the quota counter.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        credit_back(quota_subject, requested_rows)
        return None
