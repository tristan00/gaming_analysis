import requests

token = 'bae0d116ca9f488e8c268966af19d5f6be5a428a'


def get_profile():

    resp = requests.get(url='https://www.strava.com/api/v3/athlete', headers={'Authorization':f'Bearer {token}'})
    print(resp.json())

if __name__ == '__main__':
    get_profile()