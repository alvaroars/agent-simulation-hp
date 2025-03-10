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

# Import the modules
try:
    from simulator import (
        r_function,
    )
except ImportError:
    from .simulator import (
        r_function,
    )
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
        test_group_h,
        test_individual_h,
        excel_results,
        test_V,
    )
except ImportError:
    from .functions import (
        test_group_h,
        test_individual_h,
        excel_results,
        test_V,
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


class TestSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used across all tests."""
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
        """Set up test fixtures."""
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
        """Test individual heuristic performance with different heuristics and hardcoded V function."""
        # Test case 1: Basic heuristic
        r1 = test_individual_h([1, 2, 3], self.V_linear, prints=False)
        self.assertEqual(r1[0], 1)
        self.assertAlmostEqual(r1[1], 29.47, places=2)

        # Test case 2: Sequential heuristic
        h1 = [i for i in range(1, 4)]
        r2 = test_individual_h(h1, V=self.V_linear, prints=False)
        self.assertEqual(r2[0], 1)
        self.assertAlmostEqual(r2[1], 29.47, places=2)

        # Test case 3: Larger heuristic
        max_h = 6
        h2 = [i for i in range(1, max_h)]
        r3 = test_individual_h(h2, V=self.V_linear, prints=False)
        self.assertEqual(r3[0], 1)
        self.assertAlmostEqual(r3[1], 25.3, places=1)

    def test_phi_function_initialization(self):
        """Test PhiFunction initialization with different parameters."""
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
        """Test phi_region search functionality."""
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
        res = test_group_h(heuristics_k_test, self.V_linear, no_random=10, analysis=False)

        self.assertGreater(res[0][0], 0.99)  # Expected value random
        self.assertEqual(res[1], (1.0, 10.99))  # Expected value best
        self.assertIsNone(res[2])  # Intersection
        self.assertEqual(res[3], 10)  # Distinct heuristics sorted
        self.assertEqual(res[4], 1)  # Distinct heuristics expected

    def test_group_dynamics_oscillatory(self):
        """Test group dynamics with oscillatory V function."""
        # Create test data
        heuristics_k_test = [list(p) for p in permutations(range(1, l + 1), k)]
        additional_bests = [PhiFunction(V=self.V_params, region=list(range(100 - i)))
                            for i in range(10)]

        # Test group dynamics
        resp = test_group_h(heuristics_k_test, self.V_params,
                            additional_best=additional_bests,
                            analysis=False)

        self.assertIsInstance(resp[1][0], float)  # Expected value best
        self.assertIsInstance(resp[2], (int, type(None)))  # Intersection
        self.assertIsInstance(resp[3], int)  # Distinct heuristics sorted
        self.assertIsInstance(resp[4], int)  # Distinct heuristics expected

    def test_phi_group_functionality(self):
        """Test PhiGroup class functionality."""
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
        exp_value, exp_iter = group.expected_value_group()
        self.assertIsInstance(exp_value, float)
        self.assertIsInstance(exp_iter, float)

    def test_v_function_creation(self):
        """Test VFunction creation and properties."""
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
        """Test r_function and best phi calculations."""
        heuristics_k_test = [list(p) for p in permutations(range(1, l + 1), k)]

        # Get best phis
        best_phi, best_heuristics, intersection, distinct_sorted, distinct_exp = (
            self.V_r.best_phis(heuristics=heuristics_k_test, no_elements=10))

        # Create group from best phis
        best_phi_group = PhiGroup(phi_list=best_phi)

        # Test expected values
        expected_value, expected_iterations = best_phi_group.expected_value_group()
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
        test_V(params=params, n=n, l=l, k=k, trials=2, no_random=5,
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
        """Results should be independent of delta."""
        params = VFunction._random_params(n)
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
        """More tests for the results when the V function is increasing when we know the apriori some results."""
        V = self.V_linear
        heuristics = [list(p) for p in permutations(range(1, l + 1), k)]
        random_heur = random.sample(heuristics, 1)[0]
        while 1 not in random_heur:
            random_heur = random.sample(heuristics, 1)[0]
        # If heuristic contain the number one, for sure it achieve the maximum
        res = test_individual_h(random_heur, V, prints=False)
        self.assertEqual(res[0], 1)
        # Random heuristic not containing one must be worse than the one 1 as it does not achieve the maximum
        # at the penultimate point
        random_heur_not_one = random.sample(heuristics, 1)[0]
        while 1 in random_heur_not_one:
            random_heur_not_one = random.sample(heuristics, 1)[0]
        res = test_individual_h(random_heur_not_one, V, prints=False)
        self.assertLess(res[0], 1)
        # Join both and see that as a group they are perfect
        heuristics_group = [random_heur, random_heur_not_one]
        ev_random, ev_best, intersection, distinct_heur_sorted, distinct_heur_exp = test_group_h(heuristics_group, V,
                                                                                                 no_random=2,
                                                                                                 analysis=False)
        self.assertEqual(ev_best[0], 1)
        self.assertEqual(ev_random[0], 1)
        heuristics_group_2 = [random_heur_not_one] * 10
        ev_random_2, ev_best_2, intersection_2, distinct_heur_sorted_2, distinct_heur_exp_2 = test_group_h(
            heuristics_group_2, V, no_random=2, analysis=False)
        self.assertEqual(ev_best_2[0], res[0])
        self.assertEqual(ev_random_2[0], res[0])

    def test_random_V_apriori(self):
        """Test the results when the V function is random and we know apriori some results."""
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
