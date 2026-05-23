import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class ScoutFuzzyEngine:
    def __init__(self):
        # 1. Antecedents (Inputs)
        self.price = ctrl.Antecedent(np.arange(0, 101, 1), 'price_suitability')
        self.location = ctrl.Antecedent(np.arange(0, 101, 1), 'location_score')
        self.size = ctrl.Antecedent(np.arange(0, 101, 1), 'size_suitability')
        self.room_match = ctrl.Antecedent(np.arange(0, 11, 0.5), 'room_match')

        # 2. Consequent (Output)
        self.score = ctrl.Consequent(np.arange(0, 101, 1), 'suitability_score')

        self._setup_mfs()
        
        self.last_priorities = None
        self.scout_sim = None

    def _setup_mfs(self):
        # inputs membership functions remain consistent
        # p_suit always 70-100 (prices outside range are pre-filtered)
        self.price['pahali'] = fuzz.trapmf(self.price.universe, [70, 70, 75, 83])
        self.price['makul'] = fuzz.trimf(self.price.universe, [75, 83, 92])
        self.price['ucuz'] = fuzz.trapmf(self.price.universe, [87, 93, 100, 100])

        self.location['uzak'] = fuzz.trapmf(self.location.universe, [0, 0, 30, 60])
        self.location['orta'] = fuzz.trimf(self.location.universe, [40, 60, 80])
        self.location['yakin'] = fuzz.trapmf(self.location.universe, [70, 90, 100, 100])

        # size_suitability: 100 = range merkezi, 0 = range kenarı
        self.size['kucuk'] = fuzz.trapmf(self.size.universe, [0, 0, 25, 50])
        self.size['ideal'] = fuzz.trapmf(self.size.universe, [40, 70, 100, 100])

        self.room_match['uyumsuz'] = fuzz.trapmf(self.room_match.universe, [0, 0, 2, 5])
        self.room_match['kismi'] = fuzz.trimf(self.room_match.universe, [4, 6, 8])
        self.room_match['uyumlu'] = fuzz.trapmf(self.room_match.universe, [7, 9, 10, 10])

        # Output score labels
        self.score['cop'] = fuzz.trimf(self.score.universe, [0, 0, 25])
        self.score['dusuk'] = fuzz.trimf(self.score.universe, [15, 35, 55])
        self.score['orta'] = fuzz.trimf(self.score.universe, [45, 60, 75])
        self.score['yuksek'] = fuzz.trimf(self.score.universe, [65, 80, 90])
        self.score['efsane'] = fuzz.trimf(self.score.universe, [85, 100, 100])

    def get_weighted_rules(self, priorities):
        """
        priorities values: 0.0 to 1.0
        Output label for each combination is computed dynamically from the
        priority-weighted quality score, so priorities genuinely shift outcomes.
        """
        def boost(w): return w ** 1.5

        w_p = boost(priorities.get('price', 0.5))
        w_l = boost(priorities.get('location', 0.5))
        w_s = boost(priorities.get('size', 0.5))
        w_m = boost(priorities.get('rooms', 0.5))

        # Feature quality on a 0-2 scale (bad=0, neutral=1, good=2)
        P_SCORE = {'pahali': 0, 'makul': 1, 'ucuz': 2}
        L_SCORE = {'uzak': 0, 'orta': 1, 'yakin': 2}
        S_SCORE = {'kucuk': 0, 'ideal': 2}
        R_SCORE = {'uyumsuz': 0, 'kismi': 1, 'uyumlu': 2}

        def output_label(p, l, s, r):
            """Priority-weighted quality score (0-2) → output label."""
            total_w = w_p + w_l + w_s + w_m
            score = (P_SCORE[p]*w_p + L_SCORE[l]*w_l + S_SCORE[s]*w_s + R_SCORE[r]*w_m) / total_w
            if score < 0.125:  return 'cop'
            if score < 0.875:  return 'dusuk'
            if score < 1.375:  return 'orta'
            if score < 1.875:  return 'yuksek'
            return 'efsane'

        rules = []
        w = min(w_p, w_l, w_s, w_m)

        for p in ['pahali', 'makul', 'ucuz']:
            for l in ['uzak', 'orta', 'yakin']:
                for s in ['kucuk', 'ideal']:
                    for r in ['uyumsuz', 'kismi', 'uyumlu']:
                        rules.append(ctrl.Rule(
                            self.price[p] & self.location[l] & self.size[s] & self.room_match[r],
                            self.score[output_label(p, l, s, r)] % w
                        ))

        return rules

    def prepare(self, priorities):
        if self.last_priorities == priorities and self.scout_sim is not None:
            return
            
        rules = self.get_weighted_rules(priorities)
        scout_ctrl = ctrl.ControlSystem(rules)
        self.scout_sim = ctrl.ControlSystemSimulation(scout_ctrl)
        self.last_priorities = priorities.copy()

    def compute(self, inputs):
        if self.scout_sim is None:
            return 0, None

        for key, val in inputs.items():
            self.scout_sim.input[key] = val

        try:
            self.scout_sim.compute()
            return self.scout_sim.output['suitability_score'], self.scout_sim
        except:
            return 0, None
