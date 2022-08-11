import json
import glob
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import pprint

from typing import List
from sklearn.decomposition import PCA
from config import Values
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import random
from scipy import stats
from sklearn.model_selection import KFold
from helpers.timeit import timing
import numpy as np


n_estimators = 69
max_features = 0.24
max_depth = 8
model_choice = 'rf'
solver = 'saga'
penalty = 'l2'
k=5

invalid_keys = [
    'currRank',
]
all_agents_list = ['Astra',
                   'Breach',
                   'Brimstone',
                   'Chamber',
                   'Cypher',
                   'Fade',
                   'Jett',
                   'KAY/O',
                   'Killjoy',
                   'Neon',
                   'Omen',
                   'Phoenix',
                   'Raze',
                   'Reyna',
                   'Sage',
                   'Skye',
                   'Sova',
                   'Viper',
                   'Yoru']

agent_roles = {'Astra': 'Controller',
               'Breach': 'Initiator',
               'Brimstone': 'Controller',
               'Chamber': 'Sentinel',
               'Cypher': 'Sentinel',
               'Fade': 'Initiator',
               'Jett': 'Duelist',
               'KAY/O': 'Initiator',
               'Killjoy': 'Sentinel',
               'Neon': 'Duelist',
               'Omen': 'Controller',
               'Phoenix': 'Duelist',
               'Raze': 'Duelist',
               'Reyna': 'Duelist',
               'Sage': 'Sentinel',
               'Skye': 'Initiator',
               'Sova': 'Initiator',
               'Viper': 'Controller',
               'Yoru': 'Duelist'}
roles_list = ['Controller', 'Initiator', 'Duelist', 'Sentinel']
all_maps_list = ['Ascent',
                 'Bind',
                 'Breeze',
                 'Fracture',
                 'Haven',
                 'Icebox',
                 'Pearl',
                 'Split']
weapons = ['Vandal',
           'Phantom',
           'Spectre',
           'Ghost',
           'Classic',
           'Operator',
           'Sheriff',
           'Guardian',
           'Marshal',
           'Odin',
           'Judge',
           'Bulldog',
           'Stinger',
           'Frenzy',
           'Ares',
           'Shorty',
           'Bucky']
ranks = [
    'Unranked',
    'Iron 1',
    'Iron 2',
    'Iron 3',
    'Bronze 1',
    'Bronze 2',
    'Bronze 3',
    'Silver 1',
    'Silver 2',
    'Silver 3',
    'Gold 1',
    'Gold 2',
    'Gold 3',
    'Platinum 1',
    'Platinum 2',
    'Platinum 3',
    'Diamond 1',
    'Diamond 2',
    'Diamond 3',
    'Ascendant 1',
    'Ascendant 2',
    'Ascendant 3',
    'Immortal 1',
    'Immortal 2',
    'Immortal 3',
    'Radiant',
]


def get_winner(round_summary):
    return round_summary[-1]['stats']['winningTeam']['value']


def extract_team_rows(game_metadata, game_attributes, round_summary, player_summary):

    team_red_agents = list()
    team_blue_agents = list()

    map_pick = game_metadata['mapName']

    winning_team = get_winner(round_summary)

    rank_list_team_red = []
    rank_list_team_blue = []

    for i in player_summary:
        if i['metadata']['teamId'] == 'Red':
            team_red_agents.append(i['metadata']['agentName'])
            rank_list_team_red.append(i['stats']['rank']['value'])
        if i['metadata']['teamId'] == 'Blue':
            team_blue_agents.append(i['metadata']['agentName'])
            rank_list_team_blue.append(i['stats']['rank']['value'])

    row1 = create_agent_row(team_red_agents, map_pick, int('Red' == winning_team), rank_list_team_red)
    row2 = create_agent_row(team_blue_agents, map_pick, int('Blue' == winning_team), rank_list_team_blue)

    return [row1, row2]


def extract_player_rows(game_metadata, round_summary, player_summary, player_rounds_kills):
    winning_team = get_winner(round_summary)
    game_datetime = game_metadata['dateStarted']
    map_pick = game_metadata['mapName']

    data = list()

    for i in player_summary:
        new_record = dict()
        new_record['name'] = i['attributes']['platformUserIdentifier']

        agent = i['metadata']['agentName']
        agent_role = agent_roles[agent]

        new_record['game_datetime'] = game_datetime
        new_record['won_game'] = int(winning_team == i['metadata']['teamId'])
        new_record['team'] = i['metadata']['teamId']
        new_record['map_pick'] = map_pick

        for j in i['stats'].keys():
            if j in invalid_keys:
                continue
            new_record[j] = i['stats'][j]['value']

        for j in roles_list:
            if j == agent_role:
                new_record[f'role_{j}'] = 1
            else:
                new_record[f'role_{j}'] = 0

        for j in all_agents_list:
            if j == agent:
                new_record[f'agent_{j}'] = 1
            else:
                new_record[f'agent_{j}'] = 0

        weapon_kills_dict = {j: 0 for j in weapons}
        for j in player_rounds_kills:
            if 'platformInfo' in j and i['attributes']['platformUserIdentifier'] != j['platformInfo'][
                'platformUserHandle']:
                if j['metadata']['weaponName'] in weapons:
                    weapon_kills_dict[j['metadata']['weaponName']] += 1

        data.append(new_record)

    return data


def get_all_processed_data():
    files = glob.glob(r'C:\Users\trist\OneDrive\Documents\game_data\valorant_raw/*.json')

    all_records = list()
    agent_records = list()

    for file in files:

        with open(file, 'r') as f:
            json_data = json.load(f)
        json_data = json.loads(json_data)
        if 'data' not in json_data:
            print(f'error {file}')
            continue

        if json_data['data']['metadata']['modeName'] != 'Competitive':
            continue

        game_metadata = json_data['data']['metadata']
        game_attributes = json_data['data']['attributes']

        player_rounds = [i for i in json_data['data']['segments'] if i['type'] == 'player-round']
        player_rounds_damage = [i for i in json_data['data']['segments'] if i['type'] == 'player-round-damage']
        player_rounds_kills = [i for i in json_data['data']['segments'] if i['type'] == 'player-round-kills']
        player_summary = [i for i in json_data['data']['segments'] if i['type'] == 'player-summary']
        round_summary = [i for i in json_data['data']['segments'] if i['type'] == 'round-summary']
        team_summary = [i for i in json_data['data']['segments'] if i['type'] == 'team-summary']

        all_records.extend(extract_player_rows(game_metadata, round_summary, player_summary, player_rounds_kills))
        agent_records.extend(extract_team_rows(game_metadata, game_attributes, round_summary, player_summary))
    return all_records, agent_records


def create_agent_row(agent_list: List[str], map_pick: str, game_win: int, rank_list: List[str]):
    for i in agent_list:
        if i not in all_agents_list:
            raise Exception(f'Invalid agent: {i}')

    if map_pick not in all_maps_list:
        raise Exception(f'Invalid map: {map_pick}')

    row = dict()

    roles_dict = {f'role_{i}': 0 for i in roles_list}

    for i in all_agents_list:
        row['agent_' + i] = int(i in agent_list)

    for i in agent_list:
        roles_dict[f'role_{agent_roles[i]}'] += 1

    roles_dict_keys = list(roles_dict.keys())

    for i in roles_dict_keys:
        for j in roles_dict_keys:
            if roles_list.index(i.split('_')[-1]) >= roles_list.index(j.split('_')[-1]):
                continue
            roles_dict[f'{i}*{j}'] = roles_dict[i] * roles_dict[j]

    for i in all_maps_list:
        row['map_' + i] = int(i == map_pick)

    rank_avg = sum([ranks.index(i) for i in rank_list]) / max(len(rank_list), 1)
    row['rank_avg'] = rank_avg

    row.update(roles_dict)

    row['game_win'] = game_win
    return row


def predict_best_lineup(model, map_pick: str, current_agent_list: List[str], rank_list: List[str]):
    agent_tuples = list()

    for i1 in all_agents_list:
        for i2 in all_agents_list:
            for i3 in all_agents_list:
                for i4 in all_agents_list:
                    for i5 in all_agents_list:
                        if len(set([i1, i2, i3, i4, i5])) < 5:
                            continue
                        else:
                            agent_tuples.append(tuple(sorted([i1, i2, i3, i4, i5])))
    agent_tuples = list(set(agent_tuples))
    matched_agent_tuples = list()

    for i in agent_tuples:
        match = True
        for j in current_agent_list:
            if j not in i:
                match = False
        if match:
            matched_agent_tuples.append(i)

    inputs = list()
    features = list()

    for i in matched_agent_tuples:
        inputs.append(
            {'agent1_name': i[0], 'agent2_name': i[1], 'agent3_name': i[2], 'agent4_name': i[3], 'agent5_name': i[4]})
        features.append(create_agent_row(list(i), map_pick, None, rank_list))

    features_df = pd.DataFrame.from_dict(features)
    features_df_interactions = create_interactions(features_df)

    inputs_df = pd.DataFrame.from_dict(inputs)

    inputs_df.index = features_df.index

    for i in range(k):
        inputs_df[f'win_prob_{i}'] = model[i].predict_proba(features_df_interactions.drop('game_win', axis=1))[:, -1]
    inputs_df[f'win_prob'] = inputs_df[[f'win_prob_{i}' for i in range(k)]].mean(axis = 1)

    return inputs_df.sort_values('win_prob', ascending=False)


def load_vectorizer():
    with open(f'{Values.model_locations}/agent_vectorizer.pickle', 'rb') as f:
        model = pickle.load(f)
    return model


def save_vectorizer(model):
    with open(f'{Values.model_locations}/agent_vectorizer.pickle', 'wb') as f:
        pickle.dump(model, f)


def load_model():
    with open(f'{Values.model_locations}/agent_picker.pickle', 'rb') as f:
        model = pickle.load(f)
    return model

def get_model_to_train(model_choice, n_estimators, max_features, max_depth, solver, penalty, max_iter):
    if model_choice == 'rf':
        model = RandomForestClassifier(n_estimators=n_estimators, max_features = max_features, max_depth=max_depth)
    else:
        if penalty == 'elasticnet':
            model = LogisticRegression(solver=solver, penalty=penalty, l1_ratio=.5, max_iter=1000)
        else:
            model = LogisticRegression(solver=solver, penalty=penalty, max_iter=1000)
    return model


def save_model(model):
    with open(f'{Values.model_locations}/agent_picker.pickle', 'wb') as f:
        pickle.dump(model, f)


def create_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df_interaction = pd.DataFrame(index = df.index)

    df_interaction['game_win'] = df['game_win']

    columns_list = sorted(df.columns.tolist())

    for i in columns_list:
        df_interaction[i] = df[i]
        for j in columns_list:
            if  i == 'game_win' or j == 'game_win':
                continue
            if columns_list.index(i) >=columns_list.index(j):
                continue

            if 'rank' in i or 'rank' in j:
                df_interaction[f'{i}_mul_{j}'] = df[i]*df[j]
            elif 'role' in i:
                df_interaction[f'{i}_mul_{j}'] = df[i]*df[j]
                df_interaction[f'{i}_max_{j}'] = df[[i,j]].max(axis = 1)
            else:
                df_interaction[f'{i}_max_{j}'] = df[i]*df[j]

    return df_interaction


@timing
def train_model():
    all_records, all_agent_records = get_all_processed_data()
    all_agent_records_df = pd.DataFrame.from_dict(all_agent_records)

    train_df, val_df = train_test_split(all_agent_records_df, random_state = 1)


    models = list()
    preds_list = list()
    kf = KFold(n_splits=k, random_state=1, shuffle = True)

    for train_fold_np , val_fold_np in kf.split(train_df):

        train_fold_df = all_agent_records_df.iloc[train_fold_np,:]
        # val_fold_df = all_agent_records_df.iloc[val_fold_np,:]

        model = get_model_to_train(model_choice, n_estimators, max_features, max_depth, solver, penalty, 1000)
        model.fit(train_fold_df.drop('game_win', axis = 1), train_fold_df['game_win'])
        models.append(model)
        preds = model.predict_proba(val_df.drop('game_win', axis = 1))[:,-1]
        preds_list.append(preds)

    result_mode = list()
    for idx in range(val_df.shape[0]):
        tmp_result = 0
        for pred_idx in preds_list:
            tmp_result += pred_idx[idx]
        result_mode.append(tmp_result/k)


    result_mode_np = np.rint(np.array(result_mode))
    accuracy_score(result_mode_np, val_df['game_win'])
    save_model(models)



@timing
def predict(map_pick: str, current_agent_list: List[str], rank_list: List[str]) -> None:
    model = load_model()
    pp = pprint.PrettyPrinter(indent=4)
    best_results = predict_best_lineup(model=model,
                                       map_pick=map_pick,
                                       current_agent_list=current_agent_list,
                                       rank_list=rank_list)

    pp.pprint(best_results.head().to_dict(orient='records'))


if __name__ == '__main__':
    train_model()

    pp = pprint.PrettyPrinter(indent=4)

    map_pick = 'Haven'
    current_agent_list = [
    ]
    rank_list = ['Bronze 2']
    predict(map_pick=map_pick, current_agent_list=current_agent_list, rank_list=rank_list)

    rank_list = ['Radiant']
    predict(map_pick=map_pick, current_agent_list=current_agent_list, rank_list=rank_list)
