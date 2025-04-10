class OhhParseInfo:
    def __init__(self):
        self.Tournament = None
        self.Hand = None
        self.Level = None
        self.Street_id = None
        self.Round = None
        self.Pot = 0
        self.Card1 = None
        self.Card2 = None
        self.Card3 = None
        self.Card4 = None
        self.Card5 = None

        for i in range(8):
            setattr(self, f'Seat_P{i}', None)
            setattr(self, f'Dealer_P{i}', None)
            setattr(self, f'Position_P{i}', None)
            setattr(self, f'Player_P{i}', None)
            setattr(self, f'PlayerName_P{i}', None)
            setattr(self, f'Stack_P{i}', None)
            setattr(self, f'SPR_P{i}', None)
            setattr(self, f'ActionOrder_P{i}', None)
            setattr(self, f'Action_P{i}', None)
            setattr(self, f'Bet_P{i}', None)
            setattr(self, f'Allin_P{i}', None)
            setattr(self, f'Showdown_1_P{i}', None)
            setattr(self, f'Showdown_2_P{i}', None)
            setattr(self, f'PlayerWins_P{i}', None)
            setattr(self, f'WinAmount_P{i}', None)


class OhhActionInfo:
    def __init__(self):
        self.TypeBuyIn = None
        self.TournamentNumber = None
        self.StartDateUtc = None
        self.Hand = None
        self.Level = None
        self.Street_id = None
        self.Pot = 0
        self.Card1 = None
        self.Card2 = None
        self.Card3 = None
        self.Card4 = None
        self.Card5 = None
        self.Dealer = None
        self.Seat = None
        self.Round = 0
        self.ActionOrder = 0
        self.PlayerId = None
        self.PlayerName = None
        self.Stack = None
        self.SPR = None
        self.Action = None
        self.Bet = None
        self.Allin = None
        self.Showdown_1 = None
        self.Showdown_2 = None
        self.PlayerWins = None
        self.WinAmount = None
        self.Position = None

    def SetPlayerInfo(self, player_id, players_dict, seat):
        player = players_dict[player_id]
        self.Seat = player.seat
        self.Dealer = player.dealer
        self.Stack = player.stack
        self.PlayerName = player.name
        self.PlayerId = player_id
        self.Position = player.position
        pass

class Player:
    def __init__(self, id, name, seat, starting_stack, dealer_seat):
        self.id = id
        self.name = name
        self.seat = seat
        self.stack = starting_stack
        self.dealer = 1 if seat == dealer_seat else 0

class Actions:
    POST_ANTE = "Post Ante"
    POST_SB = "Post SB"
    POST_BB = "Post BB"
    FOLD = "Fold"
    RAISE = "Raise"
    CALL = "Call"
    BET = "Bet"
    CHECK = "Check"
    MUCKS_CARDS = "Mucks Cards"
    SHOW_CARDS = "Shows Cards"

class Street:
    PREFLOP = "Preflop"
    FLOP = "Flop"
    TURN = "Turn"
    RIVER = "River"
    SHOWDOWN = "Showdown"
