import requests


# https://developer.accuweather.com/accuweather-locations-api/apis/get/locations/v1/cities/search
location_api_url = 'http://dataservice.accuweather.com/locations/v1/cities/search'
r = requests.get(location_api_url)
print(r.status_code)
# 401
print(r.headers['content-type'])
# 'application/json'
print(r.encoding)
# 'utf-8'
print(r.text)
# {"fault":{"faultstring":"Failed to resolve API Key variable request.queryparam.apikey","detail":{"errorcode":"steps.oauth.v2.FailedToResolveAPIKey"}}}
print(r.json())
# {'fault': {'faultstring': 'Failed to resolve API Key variable request.queryparam.apikey', 'detail': {'errorcode': 'steps.oauth.v2.FailedToResolveAPIKey'}}}


with open("accuweather.key.txt", "r") as data_file:
    apikey = data_file.read()

params = {'apikey': apikey,
          'q': 'Barcelos',
          'language': 'pt-pt',
          'details': 'true'
          }
r = requests.get(location_api_url, params=params)
print(r.status_code)
# # 200
print(r.headers['content-type'])
# # 'application/json; charset=utf8'
print(r.encoding)
# # 'utf-8'
print(r.text)
# [{"Version":1,"Key":"272365","Type":"City",...
r_json = r.json()
print(r_json)
# [{'Version': 1, 'Key': '272365', 'Type': 'City',...

barcelos_key = r_json[0]['Key']

# https://developer.accuweather.com/accuweather-forecast-api/apis/get/forecasts/v1/daily/5day/%7BlocationKey%7D
forecast_api_url = 'http://dataservice.accuweather.com/forecasts/v1/daily/5day/' + str(barcelos_key)

params = {'apikey': apikey,
          'language': 'pt-pt',
          'details': 'true',
          'metric': 'true',
          }
r = requests.get(forecast_api_url, params=params)
print(r.text)
#
r_json = r.json()
print(r_json)
