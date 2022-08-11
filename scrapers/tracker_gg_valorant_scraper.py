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

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_driver():
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


def get_match_ids_from_userpage(user_id: str):
    logger.info(f'Getting matches for user_id: {user_id}')
    driver = get_driver()
    driver.get(f'https://tracker.gg/valorant/profile/riot/{user_id.replace("#", "%23")}/overview')
    time.sleep(10)
    soup = BeautifulSoup(driver.page_source)
    driver.close()
    links = soup.find_all('a', href=True)
    match_links = [i for i in links if 'valorant/match' in i['href']]
    return [i['href'].split('/')[-1].split('?')[0] for i in match_links]


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
    num_of_iterations=100

    for iteration in range(num_of_iterations):
        try:
            match_ids = get_all_match_ids(max_num = 10)

            user_ids = list()

            for i in match_ids:
                user_ids.extend(get_user_ids_from_match_data(i))

            user_ids = list(set(user_ids))

            random.shuffle(user_ids)
            logging.info(f'Running {len(user_ids)} user ids')

            for user_id in user_ids:
                time.sleep(10)
                match_ids = get_match_ids_from_userpage(user_id)

                logger.info(f'User: {user_id}, match_count: {len(match_ids)}' )

                for i in match_ids:
                    time.sleep(10)
                    scrape_match(i)

        except Exception as e:
            logger.exception(e)

            time.sleep(900)


if __name__ == '__main__':
    main()
