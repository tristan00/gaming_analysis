from typing import List

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
from selenium.webdriver.chrome.options import Options
import time
import sys
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import os



logging.basicConfig(format=Values.logging_format,
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_driver():
    # chrome_options = webdriver.ChromeOptions()
    # chrome_options.add_argument('--headless')
    # driver = webdriver.Chrome(Values.chromedriver_location, options=chrome_options)
    driver = webdriver.Chrome(Values.chromedriver_location)
    return driver


def scrape_match(match_id: str):
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


def get_match_ids_from_userpage(user_id: str, invalid_ranks: List[str]):

    logger.info(f'Getting matches for user_id: {user_id}')
    driver = get_driver()
    driver.get(f'https://tracker.gg/valorant/profile/riot/{user_id.replace("#", "%23")}/matches?playlist=competitive')
    time.sleep(4)

    while True:

        for i in range(100):
            driver.find_element(By.CSS_SELECTOR,'html').send_keys(Keys.DOWN)
        time.sleep(5)
        try:
            driver.find_elements(By.XPATH, "//button[contains(text(), 'Load More Matches')]")[-1].click()
        except IndexError:
            continue
        time.sleep(5)

    soup = BeautifulSoup(driver.page_source)

    for i in invalid_ranks:
        if i in driver.page_source:
            logger.info(f'Skipping user {user_id}, invalid rank: {i}')
            driver.close()
            return list()


    driver.close()
    matches = soup.find_all('div', {'class': 'match__row'})

    match_links = list()
    for match in matches:
        if 'Competitive' in str(match):
            links = match.find_all('a', href=True)
            match_links.extend([i for i in links if 'valorant/match' in i['href']])
    return list(set([i['href'].split('/')[-1].split('?')[0] for i in match_links]))


def get_user_ids_from_match_data(match_id: str):
    try:
        with open(f'{Values.valorant_data_loc}/{match_id}.json', 'r') as f:
            json_data = json.load(f)

        json_data_loaded = json.loads(json_data)

        users = list()

        if 'data' not in json_data_loaded:
            logger.info(f"error {f'{Values.valorant_data_loc}/{match_id}.json'}")
            return list()
        elif 'data' not in json_data_loaded or json_data_loaded['data']['metadata']['modeName'] != 'Competitive':
            logger.info(f"error, not competitive game,  {f'{Values.valorant_data_loc}/{match_id}.json'}")
            return list()
        else:

            for i in json_data_loaded['data']['segments']:
                if 'attributes' in i and 'platformUserIdentifier' in i['attributes']:
                    users.append(i['attributes']['platformUserIdentifier'])

            return list(set(users))
    except Exception as e:
        logger.exception(e)
        logging.warning(f'error reading match id: {match_id}')
        return list()


def get_all_match_ids(max_num: int or None = 100):
    files = glob.glob(f'{Values.valorant_data_loc}/*.json')
    if max_num and len(files) > max_num:
        files = random.sample(files, max_num)
    return [os.path.basename(i).split('.')[0] for i in files]


def main():
    num_of_iterations=1000
    invalid_ranks = ['Immortal', 'Radiant']

    for iteration in range(num_of_iterations):
        try:
            match_ids = get_all_match_ids(max_num = 1)

            user_ids = list()

            for i in match_ids:
                user_ids.extend(get_user_ids_from_match_data(i))

            user_ids = list(set(user_ids))

            random.shuffle(user_ids)

            if len(user_ids) == 0:
                user_ids.insert(0, 'Mathematics#6622')
            user_ids = ['Mathematics#6622']
            logging.info(f'Running {len(user_ids)} user ids')

            for user_id in user_ids:
                time.sleep(10)
                match_ids = get_match_ids_from_userpage(user_id, invalid_ranks)

                logger.info(f'User: {user_id}, match_count: {len(match_ids)}' )

                for i in match_ids:
                    time.sleep(10)
                    scrape_match(i)

        except Exception as e:
            logger.exception(e)

            time.sleep(900)


if __name__ == '__main__':
    main()
