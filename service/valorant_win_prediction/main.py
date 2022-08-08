import json
import glob
import pandas as pd
from sklearn.model_selection import train_test_split


def get_winner(round_summary):
    return round_summary[-1]['stats']['winningTeam']['value']

invalid_keys = [
    'currRank',
]

def extract_player_rows(game_metadata, round_summary, player_summary):
    winning_team = get_winner(round_summary)
    game_datetime = game_metadata['dateStarted']

    data = list()

    for i in player_summary:
        new_record = dict()
        new_record['name'] = i['attributes']['platformUserIdentifier']
        new_record['agent'] = i['metadata']['agentName']
        new_record['game_datetime'] = game_datetime
        new_record['won_game'] = int(winning_team == i['metadata']['teamId'])
        new_record['team'] = i['metadata']['teamId']

        for j in i['stats'].keys():
            if j in invalid_keys:
                continue
            new_record[j] = i['stats'][j]['value']

        data.append(new_record)

    return data


def get_all_processed_data():
    files = glob.glob(r'C:\Users\trist\OneDrive\Documents\game_data\valorant_raw/*.json')

    all_records = list()

    for file in files:

        with open(file, 'r') as f:
            json_data = json.load(f)
        json_data = json.loads(json_data)

        game_metadata = json_data['data']['metadata']

        player_rounds = [i for i in json_data['data']['segments'] if i['type'] == 'player-round']
        player_rounds_damage = [i for i in json_data['data']['segments'] if i['type'] == 'player-round-damage']
        player_rounds_kills = [i for i in json_data['data']['segments'] if i['type'] == 'player-round-kills']
        player_summary = [i for i in json_data['data']['segments'] if i['type'] == 'player-summary']
        round_summary = [i for i in json_data['data']['segments'] if i['type'] == 'round-summary']
        team_summary = [i for i in json_data['data']['segments'] if i['type'] == 'team-summary']


        all_records.extend(extract_player_rows(game_metadata, round_summary, player_summary))

    return all_records



def get_player_features(player_df):
    player_df = player_df.sort_values(by = ['game_datetime'])

    data = list()


    row_counter = 0
    for n, (idx, row) in enumerate(player_df.iterrows()):
        if n < 5:
            continue

        past_game = player_df.iloc[n-1:n].mean().to_dict()
        past_5_games = player_df.iloc[n-5:n].mean().to_dict()

        new_x = dict()

        for k, v in past_game.items():
            new_x[f'past_game_{k}'] = v

        for k, v in past_5_games.items():
            new_x[f'past_5_games_avg_{k}'] = v


        new_x['future_won_game'] =  row['won_game']
        data.append(new_x)


    data_df = pd.DataFrame.from_dict(data)
    return data_df


def get_features():
    all_records = get_all_processed_data()

    user_record_count = dict()

    for i in all_records:
        user_record_count.setdefault(i['name'], 0)
        user_record_count[i['name']] += 1

    user_record_count_sorted = list()

    for k, v in user_record_count.items():
        user_record_count_sorted.append({'user':k, 'count':v})

    all_records = [i for i in all_records if user_record_count[i['name']] >= 5]
    all_records_df = pd.DataFrame.from_dict(all_records)

    training_players, val_players = train_test_split(list(set(all_records_df['name'].tolist())))

    training_data_dfs = list()
    val_data_dfs = list()

    for i in training_players:

        player_df = all_records_df[all_records_df['name'] == i]
        player_data_df = get_player_features(player_df)
        training_data_dfs.append(player_data_df)


    for i in val_players:

        player_df = all_records_df[all_records_df['name'] == i]
        player_data_df = get_player_features(player_df)
        val_data_dfs.append(player_data_df)


    training_data_df = pd.concat(training_data_dfs)
    val_data_df = pd.concat(val_data_dfs)


