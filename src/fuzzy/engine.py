import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class ScoutFuzzyEngine:
    def __init__(self):
        # 1. Antecedents (Inputs)
        self.price = ctrl.Antecedent(np.arange(0, 101, 1), 'price_suitability')
        self.location = ctrl.Antecedent(np.arange(0, 11, 0.5), 'location_score')
        self.quality = ctrl.Antecedent(np.arange(0, 11, 0.5), 'listing_quality')
        self.size = ctrl.Antecedent(np.arange(0, 101, 1), 'size_suitability')
        self.llm_match = ctrl.Antecedent(np.arange(0, 11, 0.5), 'llm_alignment')

        # 2. Consequent (Output)
        self.score = ctrl.Consequent(np.arange(0, 101, 1), 'suitability_score')

        self._setup_mfs()
        
        self.last_priorities = None
        self.scout_sim = None

    def _setup_mfs(self):
        # inputs membership functions remain consistent
        self.price['pahali'] = fuzz.trapmf(self.price.universe, [0, 0, 25, 50])
        self.price['makul'] = fuzz.trimf(self.price.universe, [30, 60, 90])
        self.price['ucuz'] = fuzz.trapmf(self.price.universe, [70, 85, 100, 100])

        self.location['uzak'] = fuzz.trapmf(self.location.universe, [0, 0, 2, 5])
        self.location['orta'] = fuzz.trimf(self.location.universe, [4, 6, 8])
        self.location['yakin'] = fuzz.trapmf(self.location.universe, [7, 9, 10, 10])

        self.quality['zayif'] = fuzz.trapmf(self.quality.universe, [0, 0, 2, 4])
        self.quality['iyi'] = fuzz.trimf(self.quality.universe, [3, 6, 9])
        self.quality['mukemmel'] = fuzz.trapmf(self.quality.universe, [8, 9, 10, 10])

        self.size['kucuk'] = fuzz.trapmf(self.size.universe, [0, 0, 20, 45])
        self.size['ideal'] = fuzz.trimf(self.size.universe, [35, 65, 95])
        self.size['buyuk'] = fuzz.trapmf(self.size.universe, [80, 95, 100, 100])

        self.llm_match['uyumsuz'] = fuzz.trapmf(self.llm_match.universe, [0, 0, 2, 5])
        self.llm_match['kismi'] = fuzz.trimf(self.llm_match.universe, [4, 6, 8])
        self.llm_match['uyumlu'] = fuzz.trapmf(self.llm_match.universe, [7, 9, 10, 10])

        # Output score labels
        self.score['cop'] = fuzz.trimf(self.score.universe, [0, 0, 25])
        self.score['dusuk'] = fuzz.trimf(self.score.universe, [15, 35, 55])
        self.score['orta'] = fuzz.trimf(self.score.universe, [45, 60, 75])
        self.score['yuksek'] = fuzz.trimf(self.score.universe, [65, 80, 90])
        self.score['efsane'] = fuzz.trimf(self.score.universe, [85, 100, 100])

    def get_weighted_rules(self, priorities):
        """
        priorities values: 0.0 to 1.0
        Using power of priority to make it more dominant.
        """
        def boost(w): return w ** 1.5 # Boost the importance differences

        w_p = boost(priorities.get('price', 0.5))
        w_l = boost(priorities.get('location', 0.5))
        w_q = boost(priorities.get('quality', 0.5))
        w_s = boost(priorities.get('size', 0.5))
        w_m = boost(priorities.get('llm', 0.5))

        rules = []

        # --- Aggressive Rules for High Priorities ---
        
        # PRICE: If priority is high, "ucuz" should lead directly to Efsane/Yuksek
        rules.append(ctrl.Rule(self.price['ucuz'], self.score['efsane']))
        rules[-1].weight = w_p
        rules.append(ctrl.Rule(self.price['pahali'], self.score['cop']))
        rules[-1].weight = w_p * 1.5 # Extra penalty for expensive if price is priority

        # LOCATION: If priority is high, "yakin" should boost significantly
        rules.append(ctrl.Rule(self.location['yakin'], self.score['efsane']))
        rules[-1].weight = w_l
        rules.append(ctrl.Rule(self.location['uzak'], self.score['dusuk']))
        rules[-1].weight = w_l

        # SIZE:
        rules.append(ctrl.Rule(self.size['ideal'], self.score['yuksek']))
        rules[-1].weight = w_s
        rules.append(ctrl.Rule(self.size['kucuk'] | self.size['buyuk'], self.score['dusuk']))
        rules[-1].weight = w_s

        # QUALITY & LLM:
        rules.append(ctrl.Rule(self.quality['mukemmel'], self.score['yuksek']))
        rules[-1].weight = w_q
        rules.append(ctrl.Rule(self.llm_match['uyumlu'], self.score['efsane']))
        rules[-1].weight = w_m

        # --- Interaction / Override Rules ---
        # If Price is priority 1.0 and it's expensive, even a "yakin" location cannot save it.
        rules.append(ctrl.Rule(self.price['pahali'] & self.location['yakin'], self.score['dusuk']))
        rules[-1].weight = w_p

        # If Location is priority 1.0 and it's near, we can ignore a slightly bad price.
        rules.append(ctrl.Rule(self.location['yakin'] & self.price['pahali'], self.score['orta']))
        rules[-1].weight = w_l

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
