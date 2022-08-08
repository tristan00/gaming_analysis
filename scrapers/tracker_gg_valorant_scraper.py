import requests
from bs4 import BeautifulSoup
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
import json
from config import Values
import time
import logging
import glob
import os
import random

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_driver():
    driver = webdriver.Chrome(Values.chromedriver_location)
    return driver


def scrape_match(match_id):
    if match_id in get_all_match_ids():
        logger.info(f'Skipping id: {match_id}, already scraped')

    driver = get_driver()
    driver.get(f'https://api.tracker.gg/api/v2/valorant/standard/matches/{match_id}')
    time.sleep(3)
    json_data = driver.find_element(By.CSS_SELECTOR, 'body > pre').text

    with open(f'{Values.valorant_data_loc}/{match_id}.json', 'w') as f:
        json.dump(json_data, f)

    logger.info(f'Got data for match id: {match_id}')
    driver.close()


def get_match_ids_from_userpage(user_id):
    driver = get_driver()
    driver.get(f'https://tracker.gg/valorant/profile/riot/{user_id.replace("#", "%23")}/overview')
    time.sleep(5)
    soup = BeautifulSoup(driver.page_source)
    driver.close()
    links = soup.find_all('a', href=True)
    match_links = [i for i in links if 'valorant/match' in i['href']]
    return [i['href'].split('/')[-1].split('?')[0] for i in match_links]


def get_user_ids_from_match_data(match_id):
    with open(f'{Values.valorant_data_loc}/{match_id}.json', 'r') as f:
        json_data = json.load(f)

    json_data_loaded = json.loads(json_data)

    users = list()

    for i in json_data_loaded['data']['segments']:
        if 'attributes' in i and 'platformUserIdentifier' in i['attributes']:
            users.append(i['attributes']['platformUserIdentifier'])

    return list(set(users))


def get_all_match_ids():
    files = glob.glob(f'{Values.valorant_data_loc}/*.json')
    return [os.path.basename(i).split('.')[0] for i in files]


if __name__ == '__main__':

    match_ids = get_all_match_ids()
    user_ids = list()
    for i in match_ids:
        user_ids.extend(get_user_ids_from_match_data(i))
    user_ids = list(set(user_ids))
    random.shuffle(user_ids)

    for user_id in user_ids:
        match_ids = get_match_ids_from_userpage(user_id)

        logger.info(f'User: {user_id}, match_count: {len(match_ids)}' )

        for i in match_ids:
            time.sleep(10)
            scrape_match(i)
