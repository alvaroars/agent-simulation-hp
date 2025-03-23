"""
Test module for the simulation system.
This module contains unit tests for the simulator functionality, including
testing for V functions, Phi functions, group dynamics, and heuristic search.
It verifies the correctness of the simulation algorithms and ensures
proper functionality of all components.

Authors: Álvaro Romaniega
Version: 1.0
"""

# Import the necessary libraries
import os
import random
import unittest
from itertools import permutations
from copy import deepcopy

import sys
import pathlib
# Add parent directory to path if needed
sys.path.append(str(pathlib.Path(__file__).parent))

# Import the modules
try:
    from simulator_dynamics import (
        PhiFunction,
        PhiGroup,
        VFunction,
    )
except ImportError:
    from .simulator_dynamics import (
        PhiFunction,
        PhiGroup,
        VFunction,
    )
try:
    from functions import (
        excel_results, 
        excel_results_V,
    )
except ImportError:
    from .functions import (
        excel_results,
        excel_results_V,
    )
try:
    from config import (
        figures_path,
        results_path,
        tables_path,
    )
except ImportError:
    from .config import (
        figures_path,
        results_path,
        tables_path,
    )

# Global test variables
n = 100
l = 12
k = 3

r = {0: 0.03986917011168027, 1: 0.98, 2: 0.3520248773619988, 3: 0.06065407710421702, 4: 0.18618834631524728, 5: 0.3283724234049558, 6: 0.3274512014710405, 7: 0.12368750078701046, 8: 0.092138886186941, 9: 0.18266696938975097, 10: 0.4599928076579801, 11: 0.958, 12: 0.14465167200417575, 13: 0.5368694116670312, 14: 0.954, 15: 0.48797708107552246, 16: 0.4195326510342657, 17: 0.854, 18: 0.014550439693217964, 19: 0.43191296792611983, 20: 0.24222674987430878, 21: 0.020162688289532226, 22: 0.958, 23: 0.917, 24: 0.43, 25: 0.5916316320226339, 26: 0.07105067885834097, 27: 0.2783433026757753, 28: 0.52113079532816, 29: 0.45867509299879816, 30: 0.16774159458376894, 31: 0.533821401135266, 32: 0.3555203687176641, 33: 0.2581381886470638, 34: 0.12485646540267153, 35: 0.38239041032360616, 36: 0.16102136822000082, 37: 0.42467049860671396, 38: 0.17941595322082066, 39: 0.3543674368999485, 40: 0.0638932831609113, 41: 0.016845614031344814, 42: 0.022315191265129774, 43: 0.10202160559773818, 44: 0.976, 45: 0.012520497602619928, 46: 0.3274851816407862, 47: 0.047626564875389785, 48: 0.5525830786763045, 49: 0.807, 50: 0.17998544318823276, 51: 0.726, 52: 0.2818375500723828, 53: 0.3971410891559154, 54: 0.090105327984224, 55: 0.2630748863679214, 56: 0.5931542958942492, 57: 0.4474482462371839, 58: 0.2573785903962492, 59: 0.12312997864069176, 60: 0.09474845226494684, 61: 0.5783947221054222, 62: 0.2507401509526147, 63: 0.1926422884771502, 64: 0.5023424900153292, 65: 1.0, 66: 0.20601205936767966, 67: 0.10796754103009666, 68: 0.22271379429461602, 69: 0.2091693288988786, 70: 0.28758321322967295, 71: 0.964, 72: 0.904, 73: 0.07987900763959888, 74: 0.04495409279218874, 75: 0.12094643799917497, 76: 0.5626803891453428, 77: 0.12430659225469995, 78: 0.4595678498986017, 79: 0.4373138397058434, 80: 0.06335030989632748, 81: 0.3366127406848588, 82: 0.5611993042602212, 83: 0.46298043887308044, 84: 0.4801588231633316, 85: 0.2120500855040115, 86: 0.02404526586545024, 87: 0.5946773581139867, 88: 0.14978229859282016, 89: 0.943, 90: 0.20889662462816147, 91: 0.897, 92: 0.4979007011111523, 93: 0.5350744560323799, 94: 0.19259506887568043, 95: 0.32045828242613195, 96: 0.27160692768054673, 97: 0.909, 98: 0.2907926838490029, 99: 0.16605536946867744, 100: 0.832, 101: 0.10329576468339685, 102: 0.18084342636208492, 103: 0.5199292407477647, 104: 0.5851877656361709, 105: 0.919, 106: 0.428902679097299, 107: 0.39650010957901055, 108: 0.5241550980542772, 109: 0.01270662714630586, 110: 0.46744978078485633, 111: 0.976, 112: 0.89, 113: 0.022969096191079362, 114: 0.4197796294764604, 115: 0.05916003665074671, 116: 0.1751988590513143, 117: 0.3554754195314982, 118: 0.30767427593673496, 119: 0.40499267855261767, 120: 0.5782020666043672, 121: 0.1964224251520418, 122: 0.3574795539759845, 123: 0.18371419652700077, 124: 0.26313289652032223, 125: 0.40343873747995906, 126: 0.3172123669207037, 127: 0.53573785866471, 128: 0.42127532420601776, 129: 0.1818759711986167, 130: 0.04289197254140755, 131: 0.2417170864132425, 132: 0.5064393986668368, 133: 0.3016768087221716, 134: 0.3946217027246322, 135: 0.013464245146835974, 136: 0.46224966734070705, 137: 0.16116809455720288, 138: 0.21286697622197784, 139: 0.25444963738812937, 140: 0.938, 141: 0.14731702473421215, 142: 0.41462493021858043, 143: 0.04426936339782448, 144: 0.934, 145: 0.06980615202456388, 146: 0.3365185474412198, 147: 0.2171143129603109, 148: 0.507192783174578, 149: 0.03344862948755558, 150: 0.5316459544027619, 151: 0.0425738559673771, 152: 0.5550046048804804, 153: 0.43925872867400223, 154: 0.07869681464045779, 155: 0.44828799152801196, 156: 0.4561469603204802, 157: 0.986, 158: 0.3514373895904163, 159: 0.2845351506781543, 160: 0.44528353883977717, 161: 0.35222556477663614, 162: 0.41998244268337426, 163: 0.5332769725569151, 164: 0.40335493951256385, 165: 0.019237789188400623, 166: 0.08985418193784878, 167: 0.921, 168: 0.5596421999039916, 169: 0.35067408269796935, 170: 0.07186653426213492, 171: 0.3740766016725127, 172: 0.21568234982629744, 173: 0.5369081938746604, 174: 0.855, 175: 0.754, 176: 0.5069161942398581, 177: 0.26865314659143863, 178: 0.5865056367480456, 179: 0.667, 180: 0.84, 181: 0.4120315163210901, 182: 0.40815686468616125, 183: 0.2810011899637797, 184: 0.012729913951869775, 185: 0.061479124525322136, 186: 0.2707701997267153, 187: 0.26183079752060995, 188: 0.31204312500188197, 189: 0.3804578011044928, 190: 0.03535767610704663, 191: 0.1922823227332293, 192: 0.07434643158137615, 193: 0.36310627537806395, 194: 0.5608124662960406, 195: 0.33884648108960924, 196: 0.2548961662592731, 197: 0.04002585247725527, 198: 0.765, 199: 0.981}

def r_function(i):
    """Return the value of r for a given index i for a hardcoded r function."""
    return r[i]

class TestSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used across all tests. Only run once before all tests."""
        # Create test directories if they don't exist
        for path in [results_path, figures_path, tables_path]:
            if not os.path.exists(path):
                os.makedirs(path)

        # Store original paths for cleanup
        cls.original_paths = {
            'results': os.listdir(results_path) if os.path.exists(results_path) else [],
            'figures': os.listdir(figures_path) if os.path.exists(figures_path) else [],
            'tables': os.listdir(tables_path) if os.path.exists(tables_path) else []
        }

    def setUp(self):
        """Set up test fixtures. Run before each test."""
        self.V_linear = VFunction(n=n, function=lambda i: i / (n - 1))
        self.V_random = VFunction(n=n, random=True)
        self.V_params = VFunction(n=n, params=(0.5, 0.5, 3, 50, 0))
        self.V_r = VFunction(n=n, function=r_function)

    @classmethod
    def tearDownClass(cls):
        """Clean up any files created during testing."""
        # Remove any new files in results directory
        current_results = os.listdir(results_path)
        for file in current_results:
            if file not in cls.original_paths['results']:
                os.remove(os.path.join(results_path, file))

        # Remove any new files in figures directory
        current_figures = os.listdir(figures_path)
        for file in current_figures:
            if file not in cls.original_paths['figures']:
                os.remove(os.path.join(figures_path, file))

        # Remove any new files in tables directory
        current_tables = os.listdir(tables_path)
        for file in current_tables:
            if file not in cls.original_paths['tables']:
                os.remove(os.path.join(tables_path, file))

    def test_individual_heuristic(self):
        """Test individual heuristic performance with different heuristics and hardcoded :math:`V` function."""
        # Test case 1: Basic heuristic
        r1 = PhiFunction.test([1, 2, 3], self.V_linear, prints=False)
        self.assertEqual(r1[0], 1)
        self.assertAlmostEqual(r1[1], 29.47, places=2)

        # Test case 2: Sequential heuristic
        h1 = [i for i in range(1, 4)]
        r2 = PhiFunction.test(h1, self.V_linear, prints=False)
        self.assertEqual(r2[0], 1)
        self.assertAlmostEqual(r2[1], 29.47, places=2)

        # Test case 3: Larger heuristic
        max_h = 6
        h2 = [i for i in range(1, max_h)]
        r3 = PhiFunction.test(h2, self.V_linear, prints=False)
        self.assertEqual(r3[0], 1)
        self.assertAlmostEqual(r3[1], 25.3, places=1)

    def test_phi_function_initialization(self):
        """Test :math:`\\Phi` function initialization with different parameters."""
        # Test initialization with region
        phi_region = PhiFunction(V=self.V_linear, region=[1, 2, 3])
        self.assertEqual(phi_region.region, [1, 2, 3])

        # Test initialization with radius
        phi_radius = PhiFunction(V=self.V_linear, radius=5)
        self.assertEqual(phi_radius.radius, 5)

        # Test initialization with heuristics
        phi_heur = PhiFunction(V=self.V_linear, heuristics=[1, 2, 3])
        self.assertEqual(phi_heur.heuristics, [1, 2, 3])

        # Test error when multiple parameters provided
        with self.assertRaises(ValueError):
            PhiFunction(V=self.V_linear, region=[1, 2], radius=5)

    def test_phi_region_search(self):
        """Test :math:`\\phi` region search functionality."""
        phi = PhiFunction(V=self.V_linear, region=[1, 2, 3])

        # Test basic search
        result = phi.phi_region(0, self.V_linear)
        self.assertIsInstance(result, int)

        # Test with return_checks
        result, checks = phi.phi_region(0, self.V_linear, return_checks=True)
        self.assertIsInstance(result, int)
        self.assertIsInstance(checks, int)

    def test_phi_hp_algorithm(self):
        """Test Hong-Page algorithm implementation."""
        # Test basic HP algorithm
        result = PhiFunction.phi_hp(0, [1, 2, 3], self.V_linear)
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 0)
        self.assertLess(result, n)

        # Test with return_checks
        result, checks = PhiFunction.phi_hp(0, [1, 2, 3], self.V_linear, return_checks=True)
        self.assertIsInstance(checks, int)
        self.assertGreater(checks, 0)

    def test_group_dynamics(self):
        """Test group dynamics performance."""
        heuristics_k_test = [list(p) for p in permutations(range(1, l + 1), k)]
        res = PhiGroup.test(heuristics_k_test, self.V_linear, no_random=10, analysis=False)

        self.assertGreater(res[0][0], 0.99)  # Expected value random
        self.assertEqual(res[1], (1.0, 10.99))  # Expected value best
        self.assertIsNone(res[2])  # Intersection
        self.assertEqual(res[3], 10)  # Distinct heuristics sorted
        self.assertEqual(res[4], 1)  # Distinct heuristics expected

    def test_group_dynamics_oscillatory(self):
        """Test group dynamics with oscillatory :math:`V` function."""
        # Create test data
        heuristics_k_test = [list(p) for p in permutations(range(1, l + 1), k)]
        additional_bests = [PhiFunction(V=self.V_params, region=list(range(100 - i)))
                            for i in range(10)]

        # Test group dynamics
        resp = PhiGroup.test(heuristics_k_test, self.V_params,
                            additional_best=additional_bests,
                            analysis=False)

        self.assertIsInstance(resp[1][0], float)  # Expected value best
        self.assertIsInstance(resp[2], (int, type(None)))  # Intersection
        self.assertIsInstance(resp[3], int)  # Distinct heuristics sorted
        self.assertIsInstance(resp[4], int)  # Distinct heuristics expected

    def test_phi_group_functionality(self):
        """Test :math:`\\Phi` group class functionality."""
        # Create a group of phi functions
        phis = [
            PhiFunction(V=self.V_linear, region=[1, 2, 3]),
            PhiFunction(V=self.V_linear, radius=5),
            PhiFunction(V=self.V_linear, heuristics=[1, 2, 3])
        ]
        group = PhiGroup(phi_list=phis)

        # Test group dynamics
        result, checks = group.phi_group(0)
        self.assertIsInstance(result, int)
        self.assertIsInstance(checks, int)

        # Test expected value calculation
        exp_value, exp_iter = group.expected_value()
        self.assertIsInstance(exp_value, float)
        self.assertIsInstance(exp_iter, float)

    def test_v_function_creation(self):
        """Test :math:`V` function creation and properties."""
        # Test random V function
        v_random = VFunction(n=n, random=True)
        self.assertTrue(hasattr(v_random, 'V'))

        # Test parametric V function
        params = (0.5, 0.5, 3, 50, 0)
        v_param = VFunction(n=n, params=params)
        self.assertTrue(hasattr(v_param, 'V'))

        # Test function-based V
        v_func = VFunction(n=n, function=lambda i: i / n)
        self.assertTrue(hasattr(v_func, 'V'))

        # Test error when multiple parameters provided
        with self.assertRaises(ValueError):
            VFunction(n=n, random=True, params=params)

    def test_r_function(self):
        """Test :math:`r` function and best :math:`\\Phi` calculations."""
        heuristics_k_test = [list(p) for p in permutations(range(1, l + 1), k)]

        # Get best phis
        best_phi, best_heuristics, intersection, distinct_sorted, distinct_exp = (
            self.V_r.best_phis(heuristics=heuristics_k_test, no_elements=10))

        # Create group from best phis
        best_phi_group = PhiGroup(phi_list=best_phi)

        # Test expected values
        expected_value, expected_iterations = best_phi_group.expected_value()
        self.assertAlmostEqual(expected_value, 0.9011856361939549, places=5)
        self.assertAlmostEqual(expected_iterations, 11.94, places=2)

        phi_value, phi_iterations = best_phi_group.phi_list[0].expected_value
        self.assertAlmostEqual(phi_value, 0.8398630424629704, places=5)
        self.assertAlmostEqual(phi_iterations, 5.24, places=2)

        # Test best heuristics
        expected_heuristics = [
            [4, 7, 9], [7, 4, 9], [4, 9, 7], [4, 7, 8],
            [9, 4, 7], [9, 7, 4], [4, 6, 9], [7, 9, 4],
            [3, 4, 9], [2, 4, 9]
        ]
        self.assertEqual(best_heuristics, expected_heuristics)

        # Test other metrics
        self.assertIsNone(intersection)
        self.assertEqual(distinct_sorted, 5)
        self.assertEqual(distinct_exp, 10)

    def test_test_V_function(self):
        """Test the test_V function with various parameters."""
        # Test with parametric V
        params = (0.5, 0.5, 3, 50, 0)
        excel_results_V(params=params, n=n, l=l, k=k, trials=2, no_random=5,
               add_radial_search=False, additional_best_no=0, latex=False, open_excel=False)

        # Verify files were created
        expected_file = f"test_V_{params}_M_2_N_5_n_{n}_l_{l}_addPhi_0_search_5_less_None_noless_0_alternative_False_stop_None.xlsx"
        self.assertTrue(os.path.exists(os.path.join(results_path, expected_file)))

    def test_excel_results(self):
        """Test the excel_results function with various configurations."""
        # Test basic excel creation
        excel_results(n=n, l=l, k=k, no_Vs=2, latex=False, percentage_random=1,
                      no_random=5, add_radial_search=False, open_excel=False)
        # results_M_2_randomVperc_1_N_5_n_100_l_12_addPhi_0_search_5_less_None_noless_0_alternative_False_stop_None.xlsx
        expected_file = (f"results_M_2_randomVperc_1_N_5_n_{n}_l_{l}_addPhi_0_search_5"
                         "_less_None_noless_0_alternative_False_stop_None.xlsx")
        self.assertTrue(os.path.exists(os.path.join(results_path, expected_file)))

        # Test with latex table creation
        excel_results(n=n, l=l, k=k, no_Vs=2, latex=True, percentage_random=1,
                      no_random=5, add_radial_search=True, delta_rho=[3, 5], open_excel=False)

        expected_file_latex = (f"results_M_2_randomVperc_1_N_5_n_{n}_l_{l}"
                               "_delta_rho_[3, 5].tex")
        self.assertTrue(os.path.exists(os.path.join(tables_path, expected_file_latex)))

        # Test with ability difference analysis
        excel_results(n=n, l=l, k=k, no_Vs=2, latex=True, percentage_random=1,
                      no_random=5, analysis_ability_difference=True, open_excel=False)

        expected_file_ab = (f"results_M_2_randomVperc_1_N_5_n_{n}_l_{l}_addPhi_0_search_5"
                            "_less_None_noless_0_alternative_False_stop_None_ab_diff.tex")
        self.assertTrue(os.path.exists(os.path.join(tables_path, expected_file_ab)))

    def test_excel_results_errors(self):
        """Test error handling in excel_results function."""
        # Test error when delta_rho provided without radial search
        with self.assertRaises(ValueError):
            excel_results(n=n, l=l, k=k, no_Vs=2, add_radial_search=False,
                          delta_rho=[3, 5], open_excel=False)

        # Test handling of invalid parameters
        with self.assertRaises(ValueError):
            excel_results(n=n, l=l, k=k, no_Vs=2, params=(1, 2), function=lambda x: x, open_excel=False)

    def test_file_content_validity(self):
        """Test that created files have valid content."""
        # Create a test file
        excel_results(n=n, l=l, k=k, no_Vs=2, latex=True, percentage_random=1,
                      no_random=5, add_radial_search=False, open_excel=False)

        expected_file = (f"results_M_2_randomVperc_1_N_5_n_{n}_l_{l}_addPhi_0_search_5"
                         "_less_None_noless_0_alternative_False_stop_None.xlsx")
        file_path = os.path.join(results_path, expected_file)

        # Check file size is non-zero
        self.assertGreater(os.path.getsize(file_path), 0)

        # Check latex file if created
        tex_file = expected_file.replace('.xlsx', '.tex')
        tex_path = os.path.join(tables_path, tex_file)
        if os.path.exists(tex_path):
            with open(tex_path, 'r') as f:
                content = f.read()
                self.assertIn('\\begin{table}', content)
                self.assertIn('\\end{table}', content)

    def test_delta_invariance(self):
        """Results should be independent of :math:`\\delta`."""
        params = VFunction.random_params(n)
        params_without_delta = params[:-1]
        delta_1 = params[-1]
        delta_2 = 0
        params_with_delta_1 = tuple(list(params_without_delta) + [delta_1])
        params_with_delta_2 = tuple(list(params_without_delta) + [delta_2])
        heuristics_k_test = [list(p) for p in permutations(range(1, l + 1), k)]
        V_1 = VFunction(n=n, params=params_with_delta_1)
        V_2 = VFunction(n=n, params=params_with_delta_2)
        best_phi_1, best_heuristics_1, intersection_1, distinct_sorted_1, distinct_exp_1 = (
            V_1.best_phis(heuristics=heuristics_k_test, no_elements=10))
        best_phi_2, best_heuristics_2, intersection_2, distinct_sorted_2, distinct_exp_2 = (
            V_2.best_phis(heuristics=heuristics_k_test, no_elements=10))
        self.assertEqual(best_heuristics_1, best_heuristics_2)

    def test_linear_V_apriori(self):
        """More tests for the results when the :math:`V` function is increasing when we know the apriori some results."""
        V = self.V_linear
        heuristics = [list(p) for p in permutations(range(1, l + 1), k)]
        random_heur = random.sample(heuristics, 1)[0]
        while 1 not in random_heur:
            random_heur = random.sample(heuristics, 1)[0]
        # If heuristic contain the number one, for sure it achieve the maximum
        res = PhiFunction.test(random_heur, V, prints=False)
        self.assertEqual(res[0], 1)
        # Random heuristic not containing one must be worse than the one 1 as it does not achieve the maximum
        # at the penultimate point
        random_heur_not_one = random.sample(heuristics, 1)[0]
        while 1 in random_heur_not_one:
            random_heur_not_one = random.sample(heuristics, 1)[0]
        res = PhiFunction.test(random_heur_not_one, V, prints=False)
        self.assertLess(res[0], 1)
        # Join both and see that as a group they are perfect
        heuristics_group = [random_heur, random_heur_not_one]
        ev_random, ev_best, intersection, distinct_heur_sorted, distinct_heur_exp = PhiGroup.test(heuristics_group, V,
                                                                                                 no_random=2,
                                                                                                 analysis=False)
        self.assertEqual(ev_best[0], 1)
        self.assertEqual(ev_random[0], 1)
        heuristics_group_2 = [random_heur_not_one] * 10
        ev_random_2, ev_best_2, intersection_2, distinct_heur_sorted_2, distinct_heur_exp_2 = PhiGroup.test(
            heuristics_group_2, V, no_random=2, analysis=False)
        self.assertEqual(ev_best_2[0], res[0])
        self.assertEqual(ev_random_2[0], res[0])

    def test_random_V_apriori(self):
        """Test the results when the :math:`V` function is random and we know apriori some results."""
        n = 25
        l = 5
        k = 3
        # Random V function but first and second best separated more than 5 positions
        delta = 0
        while delta <= 5:
            j_1 = random.randint(0, n - 1)
            delta_0 = random.randint(1, n - 1)
            j_2 = (j_1 + delta_0) % n
            delta = (j_1 - j_2) % n

        def V_function(i):
            if i == j_1:
                return 1
            elif i == j_2:
                return 0.95
            else:
                return 0.94 * random.random()

        V = VFunction(n=n, function=V_function)
        heuristics = [list(p) for p in permutations(range(1, l + 1), k)]
        random_heur = deepcopy(random.sample(heuristics, 1)[
                                   0])  # Otherwise the modification of the list will affect the original one, which will affect the second test below
        for i in range(delta):
            if i not in random_heur and i != 0:
                random_heur += [i]
            phi = PhiFunction(V=V, heuristics=random_heur)
            self.assertEqual(phi(j_2), j_2)  # It cannot be j_1 as it is too far away, so second best

        random_heur += [delta]
        phi = PhiFunction(V=V, heuristics=random_heur)
        self.assertEqual(phi(j_2), j_1)  # Now it can reach j_1 as delta is reached
        # Now for a group adding phi_functions with heuristics [i]
        random_heur = random.sample(heuristics, 1)[0]
        phi_list = [PhiFunction(V=V, heuristics=random_heur)]
        for i in range(delta):
            if i not in random_heur and i != 0:
                random_heur_plus = [i]
                phi_list.append(PhiFunction(V=V, heuristics=random_heur_plus))
            group = PhiGroup(phi_list=phi_list)
            self.assertEqual(group(j_2)[0], j_2)
        random_heur_plus = [delta]
        phi_list.append(PhiFunction(V=V, heuristics=random_heur_plus))
        group = PhiGroup(phi_list=phi_list)
        self.assertEqual(group(j_2)[0], j_1)


if __name__ == '__main__':
    unittest.main()
