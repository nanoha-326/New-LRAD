import pandas as pd
import os
import json
import gspread
from google.oauth2.service_account import Credentials

def append_to_csv(question, answer, path):
    df = pd.DataFrame([{ "timestamp": pd.Timestamp.now().isoformat(), "question": question, "answer": answer }])
    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode='a', header=False, index=False)

def append_to_gsheet(question, answer, sheet_key, service_account_info):
    if isinstance(service_account_info, str):
        service_account_info = json.loads(service_account_info)
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(service_account_info, scopes=scope)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_key)
    worksheet = sh.sheet1
    worksheet.append_row([question, answer])

def load_chat_logs(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["timestamp", "question", "answer"])
    return pd.read_csv(path)
