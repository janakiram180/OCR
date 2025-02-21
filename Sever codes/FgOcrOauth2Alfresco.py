import requests
import json

CI = "fooClientId"
CS = "secret"
UN = "Admin"
PW = "Admin@123"
oAuthpart = "/oauth/token"

payload = f"grant_type=password&client_id={CI}&client_secret={CS}&username={UN}&password={PW}"
headers = {'accept': "application/json" , "Content-Type":"application/x-www-form-urlencoded"}


class AlfrescoOAuth2Client:

    def generate_token(url):
        response = requests.request("POST", url+oAuthpart, data=payload, headers=headers)
        TokenResponse = response.json()
        auth_token=TokenResponse['access_token']
        return auth_token
