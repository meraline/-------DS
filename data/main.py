import json
import logging
import os
import pandas as pd
import copy
from datetime import datetime
from models import OhhParseInfo, OhhActionInfo, Player, Actions, Street
from typing import Dict

# Use True or False
use_version_1 = True
user_version_2 = True

# массив c id "улиц" которые будут ввыведены в файл, для всех: [0, 1, 2, 3, 4]
output_street_ids = [0, 1, 2, 3, 4]

positions = {
    8: ["BTN", "SB", "BB", "EP3", "MP1", "MP2", "MP3", "CO"],
    7: ["BTN", "SB", "BB", "MP1", "MP2", "MP3", "CO"],
    6: ["BTN", "SB", "BB", "MP2", "MP3", "CO"],
    5: ["BTN", "SB", "BB", "MP2", "MP3", "CO"],
    4: ["BTN", "SB", "BB", "CO"],
    3: ["BTN", "SB", "BB"],
    2: ["SB", "BB"]
}


def assign_positions(players, dealer_seat):
    num_players = len(players)
    pos_list = positions[num_players]
    dealer_index = next(i for i, player in enumerate(players) if player['seat'] == dealer_seat)

    for i in range(num_players):
        player_index = (dealer_index + i) % num_players
        players[player_index]['position'] = pos_list[i]

    return players


# v1: parsing to one line - one round on the table
def parse_ohh_old(data_ohh):
    player_actioned_in_round = {0: set()}
    ohh = OhhParseInfo()
    ohh_results = []

    def finish_round(new_pot):
        ohh.Round = table_round[0]
        if ohh.Street_id in output_street_ids:
            ohh_results.append(copy.copy(ohh))
        global action_order
        action_order = 1
        table_round[0] += 1
        player_actioned_in_round[table_round[0]] = set()
        ohh.Pot = new_pot

        for p_i in range(8):
            p_action = getattr(ohh, f'Action_P{p_i}', None)
            p_sst = getattr(ohh, f'Stack_P{p_i}', None)
            p_bet = getattr(ohh, f'Bet_P{p_i}', None)
            if p_i in player_actioned_in_round[table_round[0] - 1] and p_action != Actions.FOLD:
                if p_sst is not None:
                    new_stack = p_sst - (p_bet if p_bet is not None else 0)
                    setattr(ohh, f'Stack_P{p_i}', new_stack if new_stack >= 0 else 0)
                setattr(ohh, f'Bet_P{p_i}', None)
            else:
                setattr(ohh, f'Bet_P{p_i}', None)
            setattr(ohh, f'Allin_P{p_i}', None)
            setattr(ohh, f'Action_P{p_i}', None)
            setattr(ohh, f'SPR_P{p_i}', None)
            if getattr(ohh, f'Player_P{p_i}') is None:
                setattr(ohh, f'ActionOrder_P{p_i}', None)
            else:
                setattr(ohh, f'ActionOrder_P{p_i}', 0 if p_i in player_actioned_in_round[1] else None)
            setattr(ohh, f'Showdown_1_P{p_i}', None)
            setattr(ohh, f'Showdown_2_P{p_i}', None)
            setattr(ohh, f'PlayerWins_P{p_i}', None)
            setattr(ohh, f'WinAmount_P{p_i}', None)

    ohh.Tournament = f"{data_ohh['tournament_info']['type']} {int(data_ohh['tournament_info']['buyin_amount'])}"
    ohh.Hand = data_ohh['game_number']
    ohh.Level = data_ohh['big_blind_amount']
    dealer_seat = data_ohh['dealer_seat']
    players = assign_positions(data_ohh['players'], dealer_seat)
    pot = 0

    for player in players:
        seat = player['seat'] - 1  # seat to be 0-based index
        setattr(ohh, f'Seat_P{seat}', player['seat'])
        setattr(ohh, f'Dealer_P{seat}', 1 if player['seat'] == dealer_seat else 0)
        setattr(ohh, f'Player_P{seat}', player['id'])
        setattr(ohh, f'PlayerName_P{seat}', player['name'])
        setattr(ohh, f'Stack_P{seat}', player['starting_stack'])
        setattr(ohh, f'Position_P{seat}', player['position'])

    rounds = data_ohh['rounds']
    table_round = [0]
    global action_order
    action_order = 1

    for r in rounds:
        ohh.Street_id = r['id']
        if r['street'] == Street.FLOP:
            ohh.Card1 = r['cards'][0]
            ohh.Card2 = r['cards'][1]
            ohh.Card3 = r['cards'][2]
        elif r['street'] == Street.TURN:
            ohh.Card4 = r['cards'][0]
        elif r['street'] == Street.RIVER:
            ohh.Card5 = r['cards'][0]

        for action in r['actions']:
            player_id = action['player_id']
            seat = [player['seat'] for player in players if player['id'] == player_id][0] - 1

            if player_id in player_actioned_in_round[table_round[0]] and action['action'] not in [Actions.POST_SB, Actions.POST_BB]:
                finish_round(pot)
            player_actioned_in_round[table_round[0]].add(player_id)

            if r['street'] != Street.SHOWDOWN:
                action_amount = action['amount']
                pot += action_amount
                round_start_stack = getattr(ohh, f'Stack_P{seat}', None)
                round_start_pot = ohh.Pot
                setattr(ohh, f'SPR_P{seat}', round(round_start_stack / round_start_pot, 2) if round_start_pot != 0 else 0)
                setattr(ohh, f'Action_P{seat}', action['action'])
                setattr(ohh, f'Allin_P{seat}', 1 if action['is_allin'] else 0)
                if action['action'] in [Actions.POST_SB, Actions.POST_ANTE]:
                    setattr(ohh, f'ActionOrder_P{seat}', 1)
                else:
                    setattr(ohh, f'ActionOrder_P{seat}', action_order)
                    action_order += 1
                if action['action'] not in [Actions.POST_SB, Actions.POST_BB]:
                    setattr(ohh, f'Bet_P{seat}', action_amount)
                else:
                    ante_amount = getattr(ohh, f'Bet_P{seat}', None)
                    setattr(ohh, f'Bet_P{seat}', (ante_amount if ante_amount is not None else 0) + action_amount)
                    if action['action'] == Actions.POST_BB:
                        finish_round(pot)
            else:
                setattr(ohh, f'ActionOrder_P{seat}', None)
                setattr(ohh, f'Action_P{seat}', action['action'])
                if action['action'] == Actions.SHOW_CARDS:
                    setattr(ohh, f'Showdown_1_P{seat}', action['cards'][0])
                    setattr(ohh, f'Showdown_2_P{seat}', action['cards'][1])
                if action['action'] == Actions.MUCKS_CARDS:
                    setattr(ohh, f'PlayerWins_P{seat}', 0)
                if any(r['street'] == Street.SHOWDOWN for r in rounds):
                    for pot in data_ohh['pots']:
                        for winner in pot['player_wins']:
                            player_id = winner['player_id']
                            win_amount = winner['win_amount']
                            seat = [player['seat'] for player in players if player['id'] == player_id][0] - 1
                            setattr(ohh, f'PlayerWins_P{seat}', 1)
                            setattr(ohh, f'WinAmount_P{seat}', win_amount)
        finish_round(pot)
    return ohh_results


# v2: parsing to one line - one player's action
def parse_ohh_new(data_ohh):
    players_dict: Dict[int, Player] = {}
    ohh = OhhActionInfo()
    ohh_results = []
    player_actioned_in_round = {0: set()}  # Keep track of players who acted in each round

    def add_action_row():
        ohh.ActionOrder += 1
        if ohh.Street_id in output_street_ids:
            ohh_results.append(copy.copy(ohh))
        if ohh.Bet is not None and ohh.PlayerId in players_dict:
            players_dict[ohh.PlayerId].stack -= ohh.Bet

    def finish_round():
        ohh.Round += 1  # Increment round at the end of the round
        ohh.ActionOrder = 0
        player_actioned_in_round[ohh.Round] = set()  # Reset players who acted in the new round

    ohh.TypeBuyIn = f"{data_ohh['tournament_info']['type']} {int(data_ohh['tournament_info']['buyin_amount'])}"
    ohh.TournamentNumber = data_ohh['tournament_info']['tournament_number']
    ohh.StartDateUtc = data_ohh['start_date_utc']
    ohh.Hand = data_ohh['game_number']
    ohh.Level = data_ohh['big_blind_amount']
    dealer_seat = data_ohh['dealer_seat']
    previous_bet = 0
    player_wins = {}

    for p in data_ohh['pots']:
        for pw in p['player_wins']:
            player_wins[pw['player_id']] = pw['win_amount']

    players = assign_positions(data_ohh['players'], dealer_seat)

    for player in players:
        players_dict[player['id']] = Player(player['id'], player['name'], player['seat'], player['starting_stack'], dealer_seat)
        players_dict[player['id']].position = player['position']

    for r in data_ohh['rounds']:
        # ohh.Round = 0
        ohh.Street_id = r['id']
        if r['street'] == Street.FLOP:
            ohh.Card1 = r['cards'][0]
            ohh.Card2 = r['cards'][1]
            ohh.Card3 = r['cards'][2]
        elif r['street'] == Street.TURN:
            ohh.Card4 = r['cards'][0]
        elif r['street'] == Street.RIVER:
            ohh.Card5 = r['cards'][0]

        for action in r['actions']:
            ohh.Pot += previous_bet
            player_id = action['player_id']

            if player_id in player_actioned_in_round[ohh.Round] and action['action'] not in [Actions.POST_SB, Actions.POST_BB]:
                finish_round()  # Move to the next round if the player has already acted
            player_actioned_in_round[ohh.Round].add(player_id)  # Mark player as having acted in this round

            ohh.SetPlayerInfo(player_id, players_dict, dealer_seat)
            ohh.SPR = round(ohh.Stack / ohh.Pot, 2) if ohh.Pot != 0 else 0
            ohh.Bet = action['amount']
            previous_bet = ohh.Bet
            ohh.Allin = 1 if action['is_allin'] else 0
            ohh.Action = action['action']
            if action['action'] == Actions.SHOW_CARDS:
                ohh.Showdown_1 = action['cards'][0]
                ohh.Showdown_2 = action['cards'][1]
            else:
                ohh.Showdown_1 = None
                ohh.Showdown_2 = None
            if action['action'] in (Actions.SHOW_CARDS, Actions.MUCKS_CARDS):
                win = player_wins.get(player_id)
                ohh.SPR = None
                ohh.PlayerWins = 1 if win is not None else 0
                ohh.WinAmount = win
            else:
                ohh.PlayerWins = None
                ohh.WinAmount = None
            add_action_row()
        finish_round()

    return ohh_results


def process_ohh_file(filename):
    ohh_results_old = []
    ohh_results_new = []
    with open(filename, 'r') as file:
        json_object_str = ''
        brace_count = 0
        for line in file:
            brace_count += line.count('{')
            brace_count -= line.count('}')
            json_object_str += line.strip()
            if brace_count == 0 and json_object_str:
                try:
                    json_object = json.loads(json_object_str)
                    ohh_object = json_object.get("ohh")
                    if ohh_object is not None:
                        if use_version_1:
                            ohh_results_old.extend(parse_ohh_old(ohh_object))
                        if user_version_2:
                            ohh_results_new.extend(parse_ohh_new(ohh_object))
                except json.JSONDecodeError as e:
                    logging.error(f"Error processing file '{filename}': {e}")
                    print(f"Invalid JSON object: {json_object_str}")
                json_object_str = ''
    return ohh_results_old, ohh_results_new


input_folder_path = "/home/tofan/Документы/GitLab_grace/ДИПЛОМ DS/data/"
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_directory = f"run_{current_time}"
output_folder_path = os.path.join(run_directory, "output")
logs_folder_path = os.path.join(run_directory, "logs")
os.makedirs(output_folder_path, exist_ok=True)
os.makedirs(logs_folder_path, exist_ok=True)

error_log_filename = os.path.join(logs_folder_path, 'error_log.txt')
success_log_filename = os.path.join(logs_folder_path, 'success_log.txt')
status_log_filename = os.path.join(logs_folder_path, 'status_log.txt')

logging.basicConfig(filename=error_log_filename, level=logging.WARNING,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

if not os.path.exists(input_folder_path):
    print(f"ERROR: Folder '{input_folder_path}' not found.")
else:
    files = [f for f in os.listdir(input_folder_path) if
             os.path.isfile(os.path.join(input_folder_path, f)) and f.endswith(".ohh")]
    if not files:
        print(f"No .ohh files found in folder '{input_folder_path}'.")
    else:
        with open(success_log_filename, 'a') as success_log_file, open(status_log_filename, 'a') as status_log_file:
            for filename in files:
                input_file_path = os.path.join(input_folder_path, filename)
                status_log_file.write(f"{filename} started\n")
                ohh_results_old, ohh_results_new = process_ohh_file(input_file_path)

                if ohh_results_old:
                    dicts_old = [x.__dict__ for x in ohh_results_old]
                    df_old = pd.DataFrame(dicts_old)
                    if not df_old.empty:
                        result_filename_old = f'parsed_{filename}_{current_time}_v1.csv'
                        output_file_path_old = os.path.join(output_folder_path, result_filename_old)
                        df_old.to_csv(output_file_path_old)
                        success_log_file.write(
                            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} File processed successfully: {filename} (old format)\n")

                if ohh_results_new:
                    dicts_new = [x.__dict__ for x in ohh_results_new]
                    df_new = pd.DataFrame(dicts_new)
                    if not df_new.empty:
                        result_filename_new = f'parsed_{filename}_{current_time}_v2.csv'
                        output_file_path_new = os.path.join(output_folder_path, result_filename_new)
                        df_new.to_csv(output_file_path_new)
                        success_log_file.write(
                            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} File processed successfully: {filename} (new format)\n")

                status_log_file.write(f"{filename} processed\n")
