import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class ScoutFuzzyEngine:
    def __init__(self):
        self._setup_universe()
        self._setup_mfs()
        self.scout_ctrl = None
        self.current_priorities = {}

    def _setup_universe(self):
        self.price     = ctrl.Antecedent(np.arange(0, 101, 1),   'price_suitability')
        self.location  = ctrl.Antecedent(np.arange(0, 11, 0.5),  'location_score')
        self.quality   = ctrl.Antecedent(np.arange(0, 11, 0.5),  'listing_quality')
        self.size      = ctrl.Antecedent(np.arange(0, 101, 1),   'size_suitability')
        self.llm_match = ctrl.Antecedent(np.arange(0, 11, 0.5),  'llm_alignment')
        self.score     = ctrl.Consequent(np.arange(0, 101, 1),   'suitability_score')

    def _setup_mfs(self):
        self.price['pahali'] = fuzz.trapmf(self.price.universe, [0, 0, 25, 50])
        self.price['makul']  = fuzz.trimf(self.price.universe,  [30, 60, 90])
        self.price['ucuz']   = fuzz.trapmf(self.price.universe, [70, 85, 100, 100])

        self.location['uzak']  = fuzz.trapmf(self.location.universe, [0, 0, 2, 5])
        self.location['orta']  = fuzz.trimf(self.location.universe,  [4, 6, 8])
        self.location['yakin'] = fuzz.trapmf(self.location.universe, [7, 9, 10, 10])

        self.quality['zayif']    = fuzz.trapmf(self.quality.universe, [0, 0, 2, 4])
        self.quality['iyi']      = fuzz.trimf(self.quality.universe,  [3, 6, 9])
        self.quality['mukemmel'] = fuzz.trapmf(self.quality.universe, [8, 9, 10, 10])

        self.size['kucuk'] = fuzz.trapmf(self.size.universe, [0, 0, 20, 45])
        self.size['ideal'] = fuzz.trimf(self.size.universe,  [35, 65, 95])
        self.size['buyuk'] = fuzz.trapmf(self.size.universe, [80, 95, 100, 100])

        self.llm_match['uyumsuz'] = fuzz.trapmf(self.llm_match.universe, [0, 0, 2, 5])
        self.llm_match['kismi']   = fuzz.trimf(self.llm_match.universe,  [4, 6, 8])
        self.llm_match['uyumlu']  = fuzz.trapmf(self.llm_match.universe, [7, 9, 10, 10])

        self.score['cop']    = fuzz.trimf(self.score.universe, [0, 0, 25])
        self.score['dusuk']  = fuzz.trimf(self.score.universe, [15, 35, 55])
        self.score['orta']   = fuzz.trimf(self.score.universe, [45, 60, 75])
        self.score['yuksek'] = fuzz.trimf(self.score.universe, [65, 80, 90])
        self.score['efsane'] = fuzz.trimf(self.score.universe, [85, 100, 100])

    def _build_rules(self):
        rules = []

        # --- Tekli kurallar (15 adet) ---
        # Her input için 3 kural: düşük→dusuk, orta→orta, yüksek→yuksek
        # Böylece her input her zaman en az bir kural ateşler

        rules.append(ctrl.Rule(self.price['ucuz'],   self.score['yuksek']))
        rules.append(ctrl.Rule(self.price['makul'],  self.score['orta']))
        rules.append(ctrl.Rule(self.price['pahali'], self.score['dusuk']))

        rules.append(ctrl.Rule(self.location['yakin'], self.score['yuksek']))
        rules.append(ctrl.Rule(self.location['orta'],  self.score['orta']))
        rules.append(ctrl.Rule(self.location['uzak'],  self.score['dusuk']))

        # kucuk/buyuk → orta: biraz dışı kötü değil, sadece ideal değil
        rules.append(ctrl.Rule(self.size['ideal'], self.score['yuksek']))
        rules.append(ctrl.Rule(self.size['kucuk'], self.score['orta']))
        rules.append(ctrl.Rule(self.size['buyuk'], self.score['orta']))

        rules.append(ctrl.Rule(self.quality['mukemmel'], self.score['yuksek']))
        rules.append(ctrl.Rule(self.quality['iyi'],      self.score['orta']))
        rules.append(ctrl.Rule(self.quality['zayif'],    self.score['dusuk']))

        rules.append(ctrl.Rule(self.llm_match['uyumlu'],  self.score['yuksek']))
        rules.append(ctrl.Rule(self.llm_match['kismi'],   self.score['orta']))
        rules.append(ctrl.Rule(self.llm_match['uyumsuz'], self.score['dusuk']))

        # --- Kombinasyon kuralları: "efsane" (3 adet) ---
        # Birden fazla kriter aynı anda iyiyse efsane tetiklenir
        rules.append(ctrl.Rule(self.price['ucuz']    & self.location['yakin'], self.score['efsane']))
        rules.append(ctrl.Rule(self.price['ucuz']    & self.size['ideal'],     self.score['efsane']))
        rules.append(ctrl.Rule(self.location['yakin'] & self.size['ideal'],    self.score['efsane']))

        # --- Kombinasyon kuralları: "cop" (2 adet) ---
        # Birden fazla kriter aynı anda kötüyse çöp tetiklenir
        rules.append(ctrl.Rule(self.price['pahali'] & self.location['uzak'],  self.score['cop']))
        rules.append(ctrl.Rule(self.price['pahali'] & self.quality['zayif'],  self.score['cop']))

        # --- Etkileşim kuralı: pahali & yakin (1 adet) ---
        # Input scaling zaten öncelikleri yansıtır; bu kural sabit bir denge noktası sağlar
        rules.append(ctrl.Rule(self.price['pahali'] & self.location['yakin'], self.score['orta']))

        return rules

    @staticmethod
    def _scale(val, universe_max, priority):
        """Input değerini kullanıcı önceliğine göre ölçekler.

        Düşük öncelik → değer evrenin nötr merkezine çekilir → kriter etkisizleşir.
        Yüksek öncelik → değer aynen korunur → kriter tam ağırlıkla skor etkiler.

        priority=1.0 → scaled = val          (değişmez)
        priority=0.5 → scaled = neutral + (val - neutral) * 0.5
        priority=0.0 → scaled = neutral      (tamamen nötr)
        """
        neutral = universe_max / 2.0
        return neutral + (val - neutral) * priority

    def prepare(self, priorities):
        """Öncelikleri saklar; ControlSystem ilk çağrıda bir kez derlenir."""
        self.current_priorities = priorities
        if self.scout_ctrl is None:
            self.scout_ctrl = ctrl.ControlSystem(self._build_rules())

    def compute(self, inputs):
        if self.scout_ctrl is None:
            return 0, None

        p = self.current_priorities
        scaled = {
            'price_suitability': self._scale(inputs['price_suitability'], 100, p.get('price',    0.5)),
            'location_score':    self._scale(inputs['location_score'],     10,  p.get('location', 0.5)),
            'listing_quality':   self._scale(inputs['listing_quality'],    10,  p.get('quality',  0.5)),
            'size_suitability':  self._scale(inputs['size_suitability'],  100,  p.get('size',     0.5)),
            'llm_alignment':     self._scale(inputs['llm_alignment'],      10,  p.get('llm',      0.5)),
        }

        # Her hesaplama için taze simulation — state kirlenmesini önler
        sim = ctrl.ControlSystemSimulation(self.scout_ctrl)
        for key, val in scaled.items():
            sim.input[key] = val

        try:
            sim.compute()
            return sim.output['suitability_score'], sim
        except Exception as e:
            import streamlit as st
            st.warning(f"Fuzzy hesaplama hatası: {e} | Scaled inputs: {scaled}")
            return 0, None
