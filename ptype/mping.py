import requests
import json
import pandas as pd


def request_mping_api_data(output_file, start_date, end_date, category="Rain/Snow"):
    api_key = input("Enter MPING API Key: ")
    reqheaders = {
        'content-type': 'https://mping.ou.edu/mping/api/v2/reports',
        'Authorization': 'Token ' + api_key
    }
    url = 'http://mping.ou.edu/mping/api/v2/reports'
    start_datetime = pd.Timestamp(start_date)
    start_date_str = start_datetime.strftime("%Y-%m-%d %H:%M:%S")
    end_datetime = pd.Timestamp(end_date)
    end_date_str = end_datetime.strftime("%Y-%m-%d %H:%M:%S")
    params = {"obtime_gt": start_date_str,
              "obtime_lt": end_date_str,
              "category": category}
    data = {}
    response = requests.get(url, params=params, headers=reqheaders)
    if response.status_code != 200:
        print(f"mPING Request Failed with code {response.status_code}")
    else:
        print("mPing Request Successful")
        data = response.json()
        with open(output_file, "w") as out_file:
            json.dump(data["results"], out_file, indent=4)
    return data


if __name__ == "__main__":
    start_date = input("Start Date: ")
    end_date = input("End Date: ")
    out_file = input("Out file: ")
    data = request_mping_api_data(out_file, start_date, end_date)