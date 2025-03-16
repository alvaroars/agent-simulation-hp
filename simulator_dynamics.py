import matplotlib as plt
import numpy as np
import os
import random
from copy import deepcopy

try:
    from .auxiliary import latex_settings
except ImportError:
    from auxiliary import latex_settings

from typing import Callable, Union
from itertools import permutations
import matplotlib.pyplot as plt


# Class to represent a phi function
# noinspection PyPep8Naming
class PhiFunction:
    """Class to represent a :math:`\\phi` function.

    :param V: The function :math:`V` to be maximized
    :type V: VFunction
    :param region: The region to search the maximizer of :math:`V`, optional (default is None). See :meth:`phi_region` for more details.
    :type region: list or None
    :param random_points: The random points to search the maximizer of :math:`V`, optional (default is None). See :meth:`phi_region` for more details.
    :type random_points: int or None
    :param radius: The radius to search the maximizer of :math:`V`, optional (default is None). See :meth:`phi_region` for more details.
    :type radius: int or None
    :param heuristics: The heuristics to search the maximizer of :math:`V`, optional (default is None). See :meth:`phi_hp` for more details.
    :type heuristics: list or None
    :param less_ability_coef: The less ability coefficient to search the maximizer of :math:`V`, optional (default is None).
    :type less_ability_coef: list or None

    .. note:: The less ability coefficient is a list of at least 2 elements, the first one is the 
        type of less ability coefficient, and the rest are the parameters associated to the type. The types are:
        
        * **displacement**: the less ability coefficient is a displacement of the function :math:`V`. The parameters are:
          
          * ``c`` = less_ability_coef[0]: the displacement parameter, an integer.
        
        * **error**: the less ability coefficient is an error in the function :math:`V`. The parameters are:
          
          * ``c`` = less_ability_coef[0]: the error parameter, a float between 0 and 1. Then, the error 
            region is a random sample of ``c*n`` points from the range ``[0, n-1]``. In this region, the function
            returns :math:`1-V(i)`.
        
        * **max_min**: the less ability coefficient is a maximum away from the real maximum. The parameters are:
          
          * ``c1`` = less_ability_coef[0]: the number of minimum values to exclude, an integer.
          * ``c2`` = less_ability_coef[1]: the number of maximum values to exclude, an integer.
          * ``c3`` = less_ability_coef[2]: the coefficient to modify :math:`V`, a float between 0 and 1. That is, for the alternative
            region, the function returns :math:`V(i)\cdot(1-c_3)+1-V(i)\cdot c_3`.
        
        * **middle**: the less ability coefficient is a middle value. The parameters are:
          
          * ``c1`` = less_ability_coef[0]: the number of middle values to include, an integer.
          * ``c3`` = less_ability_coef[-1]: the new value to assign to the middle values, a float between 0 and 1.
        
        Examples
        --------
        >>> v_func = VFunction(n=100, random=True)
        >>> # Create a phi function that searches within a fixed region
        >>> phi_region = PhiFunction(V=v_func, region=[10, 20, 30, 40, 50])
        >>> # Create a phi function that uses a heuristic search strategy
        >>> phi_heuristic = PhiFunction(V=v_func, heuristics=[1, 3, 5, 7])
        >>> # Find the maximizer starting from position 15
        >>> best_point = phi_heuristic(15)
    """

    def __init__(self, V, region=None, random_points=None, radius=None, heuristics=None,
                 less_ability_coef=None) -> None:
        global plot_count
        self.V = V  # The function is a function of V
        self.n = V.n
        n = self.n
        self.region = region
        self.random_points = random_points
        self.radius = radius
        self.heuristics = heuristics
        self.error_sample = None
        self.less_ability_coef = less_ability_coef
        if self.less_ability_coef is None:
            self.alt_V = V
        else:
            self.la_type = less_ability_coef[0]
            self.la_params = less_ability_coef[1:]
            if self.la_type == "displacement":
                # Introduce a displacement
                c = self.la_params[0]
                alt_V_u = lambda i: V(int(i - c % n))
            elif self.la_type == "error":
                c = self.la_params[0]
                error_subset = random.sample(range(n), int(c * n))
                alt_V_u = lambda i: V(i) if i not in error_subset else 1 - V(i)
            elif self.la_type == "max_min" or self.la_type == "middle":
                sorted_values = sorted([V(j) for j in range(n)])
                if self.la_type == "max_min":
                    # Maximum away from real maximum
                    c1 = self.la_params[0]  # Number of minimum values to exclude
                    c2 = self.la_params[1]  # Number of maximum values to exclude
                    c3 = self.la_params[2]  # Coefficient to multiply the maximum values
                    min_values = sorted_values[:int(c1)]
                    mins = [j for j in range(n) if V(j) in min_values]
                    if c2 > 0:
                        # Exclude the maximum value
                        # max_values = sorted_values[-int(c2*2)-1:-1]
                        max_values = sorted_values[-int(c2):]
                        maxs = [j for j in range(n) if V(j) in max_values]
                    else:
                        maxs = []
                    alt_region = mins + maxs

                    def alt_V_unnormalized(i):
                        if i in alt_region:
                            return (1 - V(i)) * c3 + V(i) * (1 - c3)
                        else:
                            return V(i)

                    # Assign 1 to the maximum values so the normalization does not modify other points
                    max_value = max([alt_V_unnormalized(j) for j in range(n)])
                    maximizer = [j for j in range(n) if alt_V_unnormalized(j) == max_value]
                    alt_V_normalized = lambda i: alt_V_unnormalized(i) if i not in maximizer else 1
                    alt_V_u = alt_V_normalized  # It is already normalized, but it is called alt_V_u to be consistent
                    assert alt_V_u(maximizer[0]) == 1
                elif self.la_type == "middle":
                    # Middle values
                    c1 = self.la_params[0]
                    c3 = self.la_params[-1]
                    median_v = sorted_values[n // 2 - int(c1):n // 2 + int(c1)]
                    alt_region = [j for j in range(n) if V(j) in median_v]

                    def alt_V_u(i):
                        if i in alt_region:
                            return c3
                        else:
                            return V(i)

                # Normalize the function
                max_value = max([alt_V_u(j) for j in range(n)])
                self.alt_V = VFunction(n=n, function=lambda i: (alt_V_u(i) / max_value))
                # Plot both functions
                if c3 > 0 and plot_count < 1:
                    plt.plot([V(j) for j in range(n)])
                    # plt.plot([alt_V_u(j) for j in range(n)])
                    plt.plot([self.alt_V(j) for j in range(n)])
                    plt.legend(["V", "Alt_V normalized"])
                    plt.show()
                    plot_count += 1
            else:
                msg = "The less ability coefficient type is not recognized"
                raise ValueError(msg)

        # Raise error if more than one region is provided or none
        no_not_none = len([z for z in [region, random_points, radius, heuristics] if z is not None])
        if no_not_none != 1:
            msg = "Only one region, random_points, radius or heuristics MUST be provided"
            raise ValueError(msg)

    def __eq__(self, other) -> bool:
        """Check if two :math:`\\phi` functions are equal. By definition if they are equal as functions."""
        results_1 = [self(i) for i in range(self.n)]
        results_2 = [other(i) for i in range(other.n)]
        return results_1 == results_2

    @staticmethod
    def expected_value_static(V, phi_function=None, h=None) -> tuple[float, float]:
        """Compute the expected value of the maximum of :math:`V` using the uniform distribution.

        :return: Expected value of the maximum of :math:`V` and the average number of evaluations of :math:`V`
        :rtype: tuple[float, float]
        """
        if phi_function is None:
            phi_function = PhiFunction(V=V, heuristics=h)
        elif h is not None and phi_function is None:
            raise ValueError("Both phi_function and h cannot be provided")
        n = V.n
        values = [phi_function(i, return_checks=True) for i in
                  range(n)]  # phi is called with the return_checks flag, once
        expected_value = sum([V(values[i][0]) for i in range(n)]) / n
        expected_value_iterations = sum([values[i][1] for i in range(n)]) / n
        return expected_value, expected_value_iterations

    @property
    def expected_value(self) -> tuple[float, float]:
        return self.expected_value_static(phi_function=self.phi_function, V=self.V)

    def phi_region(self, i, V, return_checks=False) -> tuple[int, int]:
        """Compute :math:`\\phi(i)`  where :math:`\\phi(i)` is the maximizer of :math:`V` in the search space.

        :param i: The starting point.
        :type i: int
        :param V: The function :math:`V` to be maximized.
        :type V: VFunction
        :param return_checks: Flag to return the number of checks. If it is True, the function returns a tuple with the
            maximizer and the number of checks.
        :type return_checks: bool
        :return: :math:`\\phi(i)` according to the region, random_points or radius, and optionally the number of checks
        :rtype: float or tuple.

        .. note:: The search region always includes the current point `i`, so it always improves. Also, to avoid a perfect algorithm, if the search region includes the global maximizer (except it is `i` or we are taking a random sample), it returns the second best. 
            Otherwise, it would always find the global maximizer.  See also :meth:`VFunction.create_additional_bests`, where the maximizer point is not included in the argument of the phi function.
        """
        region = self.region
        radius = self.radius
        random_points = self.random_points
        n = self.n
        no_not_none = len([z for z in [region, random_points, radius] if z is not None])
        if no_not_none > 1:
            raise ValueError("Both region, random_points or radius cannot be provided")
        elif region is not None:
            search_region = region + [i]  # Add the current element to the region, so it always improves
        elif random_points is not None:
            search_region = random.sample(range(n), random_points) + [i]
        elif radius is not None:
            if isinstance(radius, int):
                search_region = list(range(i - radius, i + radius + 1))
            elif isinstance(radius, list):
                delta = radius[0]
                rad = radius[1]
                try:
                    delta_2 = radius[2]
                except IndexError:
                    delta_2 = None
                try:
                    rad_2 = radius[3]
                except IndexError:
                    rad_2 = None
                if delta_2 is None and rad_2 is None:
                    search_region = range(i - rad, i + rad + 1, delta)
                    # This is NOT the same as the one below with delta_2 and rad_2 equal to delta and rad
                elif delta_2 is not None and rad_2 is not None:
                    search_region = list(range(i - rad, i, delta)) + list(range(i, i + rad_2 + 1, delta_2))
                else:
                    raise ValueError("The radius must be an integer or a list of two elements")
            else:
                raise ValueError("The radius must be an integer or a list of two elements")
            search_region = [j % n for j in search_region] + [i]
        else:
            search_region = range(n)  # Initial region is all the elements

        # If the global maximizer is in the region, it will always be chosen, resulting in a perfect algorithm. To avoid
        # this, introduce an error so that for some proportion of points (not random), it returns the second best.
        # See also _create_additional_bests, where the maximizer is avoided
        if region is not None:
            if self.error_sample is None:
                max_value = max([V(j) for j in range(n)])
                global_maximizer = [j for j in range(n) if V(j) == max_value]
                error_sample_cond = global_maximizer[0] in search_region and i != global_maximizer[0]
                if error_sample_cond:
                    # Deterministic region where the function return a second best. For instance, take the multiples of a number
                    error_sample = [z for z in search_region if z % 10 == 0]
                else:
                    error_sample = []
                self.error_sample = error_sample
            else:
                error_sample = self.error_sample
        else:
            self.error_sample = []
        # Locate the maximizer of V in the region
        max_value_reg = max([V(j) for j in search_region])
        maximizer = [j for j in search_region if V(j) == max_value_reg]
        if self.error_sample != [] and i in self.error_sample:
            # Select the second best
            second_max_value = max([V(j) for j in search_region if V(j) != max_value_reg])
            maximizer = [j for j in search_region if V(j) == second_max_value]

        if V(maximizer[0]) < V(i):
            msg = "The phi function is not working properly"
            raise ValueError(msg)

        if return_checks:  # Only needs to check the search region one time, say the first time
            no_checks = 0
            if i == 0:
                no_checks += len(search_region)
            if i in search_region:
                no_checks += 1

            return maximizer[0], no_checks
        return maximizer[0]

    @staticmethod
    def phi_hp(i, h, V, return_checks=False) -> tuple[int, int] or int:
        """Compute :math:`\\phi(i)` using the Hong-Page algorithm.

        :param i: Starting point
        :type i: int
        :param h: Heuristic, list of :math:`k` elements
        :type h: list
        :param V: Function :math:`V` to be maximized
        :type V: VFunction
        :param return_checks: Flag to return the number of checks
        :type return_checks: bool

        :return: :math:`\\phi(i)` according to the heuristic h, and optionally the number of checks
        :rtype: float or tuple

        .. note:: The number of checks is not the number of times the heuristic is applied, but the 
            number of times the function V is evaluated.
        """
        n = V.n
        x = i
        heur = deepcopy(h)
        number_of_checks = 0
        while True:
            # Iterate through the heuristic list
            for j in range(len(heur)):
                number_of_checks += 1
                x_j = (x + heur[j]) % n
                if V(x) < V(x_j):
                    heur = heur[j + 1:] + heur[:j + 1]
                    x = x_j
                    break
                    # The break statement exits the for loop, causing the while loop to start over with the new values.
            else:
                # If no break occurred, the condition V(x) >= V(x_j) is satisfied
                if V(x) < V(i):
                    raise ValueError("The Hong-Page algorithm is not working properly")
                if return_checks:
                    return x, number_of_checks
                return x

    @staticmethod
    def from_h_to_phi(h, V) -> Callable[[int, bool], Union[tuple[int, int], int]]:
        """Convert a heuristic to a :math:`\\phi` function.

        :param h: Heuristic, list of :math:`k` elements
        :type h: list
        :param V: Function :math:`V` to be maximized
        :type V: VFunction
        :return: :math:`\\phi` function according to HP model
        :rtype: function
        """
        return lambda i, return_checks=False: PhiFunction.phi_hp(i, h, V, return_checks)

    def phi_function(self, i, return_checks=False) -> tuple[int, int] or int:
        """Compute :math:`\\phi(i)` using the region, random_points, radius or heuristics, see :meth:`phi_region` and :meth:`phi_hp` for more details.

        :param i: The starting point
        :type i: int
        :param return_checks: Flag to return the number of checks
        :type return_checks: bool
        :return: :math:`\\phi(i)` according to the region, random_points, radius or heuristics, and optionally the number of checks
        :rtype: float or tuple
        """
        if self.region is not None or self.random_points is not None or self.radius is not None:
            return self.phi_region(i, self.V, return_checks=return_checks)
        elif self.heuristics is not None:
            return self.phi_hp(i, self.heuristics, self.alt_V, return_checks)
        else:
            raise ValueError("No region, random_points, radius or heuristics provided")

    def __call__(self, i, return_checks=False) -> tuple[int, int] or int:
        """Compute :math:`\\phi(i)` using the region, random_points, radius or heuristics, see :meth:`phi_region` and :meth:`phi_hp` for more details.
        
        :param i: The starting point
        :type i: int
        :param return_checks: Flag to return the number of checks
        :type return_checks: bool
        :return: :math:`\\phi(i)` according to the region, random_points, radius or heuristics, and optionally the number of checks
        :rtype: float or tuple
        """
        return self.phi_function(i, return_checks=return_checks)

    @staticmethod
    def test(h, V, prints=False) -> tuple[float, float]:
        """Test an individual heuristic.

        :param V: The :math:`V` function.
        :type V: VFunction
        :param prints: Whether to print the results.
        :type prints: bool

        :return: The expected value of the heuristic.
        :rtype: tuple[float, float]
        """
        phi = PhiFunction(V=V, heuristics=h)
        if prints:
            print("Heuristic: ", h)
            print("Expected value: ", phi.expected_value)
            print("")
        else:
            return phi.expected_value


class PhiGroup:
    """Class to represent a group of phi functions."""

    def __init__(self, phi_list=None, is_best=None):
        self.phi_list = phi_list
        self.is_best = is_best

    def __call__(self, i, alternative_stop=False, stop=None) -> tuple[int, int] or int:
        """Compute :math:`\\phi^\\Phi(i)` using the group dynamics with a sequential approach, also known as ``relay dynamics``.

        :param i: Starting point
        :type i: int
        :param alternative_stop: (optional, default is False) Flag to stop the algorithm when there is
            a disagreement cycle
        :type alternative_stop: bool
        :param stop: (optional, default is None) The number of checks to stop the algorithm
        :type stop: int
        :return: :math:`\\phi^\\Phi(i)` according to the group dynamics and the number of checks
        :rtype: float or tuple
        """
        return self.phi_group(i, alternative_stop=alternative_stop, stop=stop)

    def __eq__(self, other) -> bool:
        """Check if two :math:`\\Phi` groups are equal. By definition if they are equal as functions."""
        n = self.phi_list[0].n
        results_1 = [self(i) for i in range(n)]
        results_2 = [other(i) for i in range(n)]
        return results_1 == results_2

    def phi_group(self, i, alternative_stop=False, stop=None, method="relay") -> tuple[int, int] or int:
        """Compute :math:`\\phi^\\Phi(i)` using the group dynamics.

        :param i: Starting point
        :type i: int
        :param alternative_stop: (optional, default is False) Flag to stop the algorithm when there is
            a disagreement cycle
        :type alternative_stop: bool
        :param method: (optional, default is "relay") The method to use to compute the group dynamics
        :type method: str
        :return: :math:`\\phi^\\Phi(i)` according to the group dynamics and the number of checks
        :rtype: tuple
        """
        if method == "relay":
            return self.phi_group_relay(i, alternative_stop=alternative_stop, stop=stop)
        else:
            raise NotImplementedError(f"Method {method} not implemented")

    def phi_group_relay(self, i, alternative_stop=False, stop=None) -> tuple[int, int] or int:
        """Compute :math:`\\phi^\\Phi(i)` using the group dynamics with a sequential approach, also known as `relay dynamics`.

        :param i: Starting point
        :type i: int
        :param phi_list: List of functions to compute phi(i) for each heuristic
        :type phi_list: list
        :param alternative_stop: (optional, default is False) Flag to stop the algorithm when there is
            a disagreement cycle
        :type alternative_stop: bool
        :param stop: (optional, default is None) The number of checks to stop the algorithm
        :type stop: int
        :return: :math:`\\phi(i)` according to the group dynamics and the number of checks
        :rtype: tuple
        """
        x = i
        if stop is None:  # No stop provided, set an "infinite" number of checks
            stop = 1e6
        group_check = 0
        # Check all share the same V
        V = self.phi_list[0].V
        for phi in self.phi_list:
            if phi.V != V:
                raise ValueError("All phi functions must share the same V function")
        list_x = [i]
        while True:
            if alternative_stop and x in list_x[:-1]:
                print("Alternative stop activated: a disagreement cycle has been detected")
                return x, group_check

            # Iterate through the list of phi functions
            for pos, phi in enumerate(self.phi_list):
                group_check += 1
                if group_check >= stop:
                    print("Alternative stop activated: too many checks")
                    return x, group_check
                if phi(x) != x:
                    if V(phi(x)) < V(x):
                        print("Disagreement detected")
                        return x, group_check  # Could be commented, so a whole cycle is needed
                    x = phi(x)
                    self.phi_list = self.phi_list[pos:] + self.phi_list[:pos]
                    list_x.append(x)
                    break
            else:
                # If no break occurred, all phi functions agree
                if V(x) < V(i) and all([phi.less_ability_coef is None for phi in self.phi_list]):
                    raise ValueError("The group dynamics is not working properly")
                return x, group_check

    def expected_value(self, alternative_stop=False, stop=None) -> tuple[float, float]:
        """Compute the expected value of the maximum of :math:`V` using the group dynamics.

        :param alternative_stop: (optional, default is False) Flag to stop the algorithm when there is
            a disagreement cycle
        :type alternative_stop: bool
        :param stop: (optional, default is None) The number of checks to stop the algorithm
        :type stop: int
        :return: Expected value of the maximum of :math:`V` and the average number of iterations
        :rtype: tuple
        """
        # Check that all phi share the same V
        V = self.phi_list[0].V
        n = V.n
        for phi in self.phi_list:
            if phi.V != V:
                raise ValueError("All phi functions must share the same V function")
        values = [self.phi_group(i, alternative_stop=alternative_stop, stop=stop) for i in range(n)]
        expected_values = sum([V(values[i][0]) for i in range(n)]) / n
        expected_iterations = sum([values[i][1] for i in range(n)]) / n
        return expected_values, expected_iterations

    def analyze_phi_region(self) -> None:
        """Analyze the :math:`\\phi` functions to check duplicates, expected values and regions.
        """
        # Check if there are duplicates in the best phi functions. Find that sorted regions are the same
        i = 0
        if len(self.phi_list) == 0:
            raise ValueError("No phi functions provided")
        V = self.phi_list[0].V
        for phi in self.phi_list:
            if phi.V != V:
                raise ValueError("All phi functions must share the same V function")
        # Print values of phi
        for i, phi in enumerate(self.phi_list):
            print("Phi ", i, " ", phi.region)
            print("V at region ", i, " ", [V(i) for i in phi.region])
            print("Expected value ", i, " ", phi.expected_value[0])

        print("Checking for duplicates in regions and expected values--------------------------------")
        for j, phi in enumerate(self.phi_list):
            for k, phi2 in enumerate(self.phi_list):
                if phi != phi2 and sorted(phi.region) == sorted(phi2.region):
                    print("Duplicate regions")
                    i = +1
                if phi.expected_value[0] == phi2.expected_value[0] and j < k:
                    print(f"Duplicate expected values ({phi.expected_value[0]})for position ", j, " and ", k)
                    # Print regions
                    print("Region ", j, " ", phi.region)
                    print("V at region ", j, " ", [V(i) for i in phi.region])
                    print("Region ", k, " ", phi2.region)
                    print("V at region ", k, " ", [V(i) for i in phi2.region])
                    # Print common values and its evaluation with V
                    common_values = [i for i in phi.region if i in phi2.region]
                    print("Common values ", common_values)
                    print("V at common values ", [V(i) for i in common_values])
                    # Now check if the functions are the same
                    n = phi.n
                    for i in range(n):
                        if phi(i) != phi2(i):
                            print("Different functions")
                            break
                    else:
                        print("Same functions. The values are: ", [phi(i) for i in range(n)])
        if i == 0:
            print("No duplicate regions")

        if len(self.phi_list) == 11:  # Hardcoded for the example of 10 additional best phi functions
            print("Expected value of the radial search: ", self.phi_list[-1].expected_value[0])

    @staticmethod
    def create_random_phi(V, random_sample, less_ability_coef=None, no_less=0, delta_rho=None) -> list[PhiFunction]:
        """
        Create random :math:`\\phi` functions from a random sample of heuristics.

        :param V: The function V to be maximized
        :type V: VFunction
        :param random_sample: The random sample of heuristics to be used for obtaining the random phi 
            functions
        :type random_sample: list
        :param less_ability_coef: (optional, default is None) The less ability coefficient to be used 
            for obtaining the random phi functions
        :type less_ability_coef: float
        :param no_less: (optional, default is 0) The number of times the less ability coefficient is 
            used
        :type no_less: int
        :param prints: (optional, default is True) Flag to print the random phi functions
        :type prints: bool
        :param delta_rho: (optional, default is None) The delta_rho to be used. If None, the random sample 
            is meant to be a list of heuristics. If not None, the random sample is meant to be a list of 
            tuples with the delta_rhos.
        :type delta_rho: list
        :return: The random phi functions
        :rtype: list of :class:`PhiFunction`
        """
        if delta_rho is None:
            type_random = "heuristics"
        else:
            type_random = "delta_rho"
        phi_random = []
        # Intercalate the less ability coefficient no times ([less_ability_coef] + [None]) no_less times
        less_ability_list = ([less_ability_coef] + [None]) * no_less + [None] * (len(random_sample)
                                                                                 - 2 * no_less)
        for j, h in enumerate(random_sample):
            if type_random == "heuristics":
                phi_random += [PhiFunction(V=V, heuristics=h, less_ability_coef=less_ability_list[j])]
            elif type_random == "delta_rho":
                phi_random += [PhiFunction(V=V, radius=h, less_ability_coef=less_ability_list[j])]

        return phi_random

    def plot_comparison(self, phi_group_alt, alternative_stop=False, stop=None, title=None,
                        name=None, sizes=None, figures_path=None) -> None:
        """Plot the :math:`\\phi` functions."""
        if sizes is not None:
            labelsize = sizes[0]
            title_size = sizes[1]
            ysize = sizes[2]
            xsize = sizes[3]
            legend_size = sizes[4]
            figsize = sizes[5]
        else:
            labelsize = 15
            title_size = 16
            ysize = 15
            xsize = 15
            legend_size = 15
            figsize = (10, 6)
        latex_settings(plt, labelsize=labelsize, figsize=figsize)
        # Red colors for best group, blues if not
        if self.is_best:
            color = "red"
        else:
            color = "blue"
        if phi_group_alt.is_best:
            if color == "red":
                # Different red color for the best group    
                color_alt = "tab:red"
            else:
                color_alt = "red"
        else:
            if color == "blue":
                color_alt = "tab:blue"
            else:
                color_alt = "blue"
        n = self.phi_list[0].n
        plt.plot([self(i, alternative_stop=alternative_stop, stop=stop)[0] for i in range(n)], color=color)
        plt.plot([phi_group_alt(i, alternative_stop=alternative_stop, stop=stop)[0] for i in range(n)], color=color_alt)
        # Horizontal, not vertical, position of the ylabel and add a separation between the ylabel and the plot
        plt.ylabel(r"$\phi^\Phi$", fontsize=ysize, rotation=0, labelpad=10)
        # Plot identity line in black
        plt.plot([i for i in range(n)], [i for i in range(n)], color="black")
        plt.xlabel(r"$x$", fontsize=xsize)
        plt.title(title, fontsize=title_size)

        # Labels
        def label_function(phi_group):
            if phi_group.is_best:
                return r"$\phi^{\Phi_B}$"
            else:
                return r"$\phi^{\Phi_R}$"

        labels = [label_function(phi_group) for phi_group in [self, phi_group_alt]]
        if labels[0] == labels[1]:
            # Add a prime to the label
            labels[1] = labels[1].replace(r"^\Phi", r"^\Phi'")
        plt.legend(labels, fontsize=legend_size)
        if name is not None:
            if figures_path is None:
                # Define a figures directory relative to the current working directory and ensure it exists
                figures_path = os.path.join(os.getcwd(), "figures")
                os.makedirs(figures_path, exist_ok=True)
            saving_path = os.path.join(figures_path, name)
            plt.savefig(saving_path)

        else:
            plt.show()
        plt.close()

    def analyze_ability_difference(self) -> tuple:
        """Analyze the ability difference between the :math:`\\phi` functions."""
        # Compute individual ability
        phis = self.phi_list
        abilities = [phi.expected_value[0] for phi in phis]
        # Compute statistics
        mean_ability = np.mean(abilities)
        min_ability = np.min(abilities)
        max_ability = np.max(abilities)
        return mean_ability, min_ability, max_ability

    @staticmethod
    def create_pool_tools(delta_rho_ratio=None, repeat=1, l=None, k=None):
        """
        Create a pool of heuristics or :math:`\\delta` and :math:`\\rho` for the group dynamics.
        
        :param delta_rho_ratio: The ratio of :math:`\\delta` to :math:`\\rho`. If None, the pool 
            of heuristics is created. If not None, the pool of :math:`\\delta` and :math:`\\rho` is created.
        :type delta_rho_ratio: float or list
        :param repeat: The number of times to repeat the pool of heuristics or :math:`\\delta` and :math:`\\rho`.
        :type repeat: int
        :param l: The number of heuristics to be used.
        :type l: int
        :param k: The number of heuristics to be used.
        :type k: int
        :return: The pool of heuristics or :math:`\\delta` and :math:`\\rho`.
        :rtype: list
        """
        if delta_rho_ratio is None:
            pool_tools = [list(p) for p in permutations(range(1, l + 1), k)]
        else:
            if isinstance(delta_rho_ratio, list):
                delta_rho_min = delta_rho_ratio[0]
                delta_rho_max = delta_rho_ratio[1]
            else:
                delta_rho_min = 1
                delta_rho_max = delta_rho_ratio
            # Pairs of deltas and rhos such that rho/delta < 3, repeat the same heuristic 100
            pool_tools = [[a, b, c, d] for a in range(1, delta_rho_max)
                          for c in range(1, delta_rho_max)
                          for b in range(1, delta_rho_max ** 2)
                          for d in range(1, delta_rho_max ** 2)
                          if b / a <= delta_rho_max and d / c <= delta_rho_max
                          and b / a >= delta_rho_min and d / c >= delta_rho_min]
            pool_tools = [pair for pair in pool_tools if pair[0] <= pair[1] and pair[2] <= pair[3]]
            pool_tools = repeat * pool_tools

        return pool_tools

    @staticmethod
    def test(heuristics, V, no_random=10, no_elements=None, additional_best=None, analysis=False,
                    less_ability_coef=None, no_less=0, alternative_stop=False, stop=None, delta_rho=None,
                    analysis_ability_difference=False) -> tuple: 
        """Test the group dynamics of a set of heuristics.

        :param heuristics: The heuristics to test.
        :type heuristics: list
        :param V: The :math:`V` function.
        :type V: VFunction
        :param no_random: The number of random heuristics to test.
        :type no_random: int
        :param no_elements: The number of elements to test.
        :type no_elements: int
        :param additional_best: The additional best heuristics to test.
        :type additional_best: list
        :param analysis: Whether to analyze the results.
        :type analysis: bool
        :param less_ability_coef: The less ability coefficient.
        :type less_ability_coef: float
        :param no_less: The number of less ability elements.
        :type no_less: int
        :param alternative_stop: Whether to use an alternative stop condition.
        :type alternative_stop: bool
        :param stop: The stop condition.
        :type stop: int
        :param delta_rho: List of delta ratios for the group dynamics.
        :type delta_rho: list
        :param analysis_ability_difference: Whether to analyze the ability difference.
        :type analysis_ability_difference: bool

        :return: The expected value of the heuristics.
        :rtype: tuple
        """

        n = V.n
        # l is the maximum number that each element in the heuristic can take
        l = max(max(heuristics))
        # k is the number of elements in the heuristic
        k = len(heuristics[0])

        if no_elements is None:
            no_elements = no_random
        random_pool_h = heuristics
        best_pool_phi = [PhiFunction(V=V, heuristics=h) for h in random_pool_h]
        random_heur = random.sample(random_pool_h, no_random)
        phi_random = PhiGroup.create_random_phi(V, random_heur, less_ability_coef=less_ability_coef,
                                                no_less=no_less, delta_rho=delta_rho)
        # Additional best phi functions, which are not included in the random pool, although highly unlikely if they were
        if additional_best is not None:
            best_pool_phi += additional_best

        # Best phis
        best_phi, best_heuristics, intersection, distinct_heur_sorted, distinct_heur_exp = (
            V.best_phis(heuristics=heuristics, no_elements=no_elements, additional_best=additional_best,
                        delta_rho=delta_rho))
        random_group = PhiGroup(phi_list=phi_random, is_best=False)
        best_group = PhiGroup(phi_list=best_phi, is_best=True)
        ev_random = random_group.expected_value(alternative_stop=alternative_stop, stop=stop)
        ev_best = best_group.expected_value(alternative_stop=alternative_stop, stop=stop)
        if analysis_ability_difference:
            ab_stats_best = best_group.analyze_ability_difference()
            ab_stats_random = random_group.analyze_ability_difference()
        if analysis:
            additional_best_no = len(additional_best) if additional_best is not None else 0
            if additional_best_no > 0 and additional_best[0].region is not None:
                search = len(additional_best[0].region)
            else:
                search = 5  # Default search
            name = f"phi_comparison_trial_V_random"
            name += f"_N_{no_random}_n_{n}_l_{l}_addPhi_{additional_best_no}_search_{search}_"
            name += f"less_{less_ability_coef}_noless_{no_less}_alternative_{alternative_stop}_"
            name += f"stop_{stop}.pdf"
            best_group.plot_comparison(random_group, title=r"$\Phi_B$ vs $\Phi_R$",
                                    name=name)
            V.plot(name="plot_V_" + name)
            print("Random group vs Best group")
            if delta_rho is None:
                for i in range(n):
                    print(
                        f"{i} -> {(random_group(i)[0], f'{V(random_group(i)[0]):.3f}')} vs {(best_group(i)[0], f'{V(best_group(i)[0]):.3f}')}")
            best_message = "Best heuristics: " + str(best_heuristics)
            if intersection is not None:
                best_message += f" and {intersection} additional best phi functions"
            print("Distinct in best heuristics: ", distinct_heur_sorted, " Distinct in expected values: ",
                distinct_heur_exp)
            print("Random heuristics: ", random_heur)
            # Extract the different numbers in the random heuristics and the missing ones
            random_heur_numbers = set([i for h in random_heur for i in h])
            missing_heur = [i for i in range(1, l + 1) if i not in random_heur_numbers]
            print("Random heuristics unique: ", random_heur_numbers)
            print("Missing heuristics: ", missing_heur)
            # The same for the best heuristics
            best_heur_numbers = set([i for h in best_heuristics for i in h])
            missing_best_heur = [i for i in range(1, l + 1) if i not in best_heur_numbers]
            print("Best heuristics unique: ", best_heur_numbers)
            print("Missing best heuristics: ", missing_best_heur)
            print(best_message)
            print("Random expected value: ", ev_random)
            print("Random expected value, individual: ", [phi.expected_value[0] for phi in phi_random])
            print("Best expected value: ", ev_best)
            print("Best expected value, individual: ", [phi.expected_value[0] for phi in best_phi])
            print("Best heuristics expected value, individual: ",
                [PhiFunction(heuristics=h, V=V).expected_value[0] for h in best_heuristics])
            print("")
            # input("Press Enter to continue...")
        if analysis_ability_difference:
            return ev_random, ev_best, intersection, distinct_heur_sorted, distinct_heur_exp, ab_stats_best, ab_stats_random
        else:
            return ev_random, ev_best, intersection, distinct_heur_sorted, distinct_heur_exp

class VFunction:
    """Class to represent a function :math:`V`.

    
    :param params: The parameters for the parametric form of the function :math:`V` according to the formula 
        in the paper. Optional, default is None.
    :type params: tuple
    :param random: Flag to create a random :math:`V` function. Optional, default is False.
    :type random: bool
    :param function: The function to be used for the :math:`V` function. Optional, default is None.
    :type function: function    
    """

    def __init__(self, n, params=None, random=None, function=None) -> None:
        self.params = params
        self.random = random
        self.function = function
        not_none = [params is not None, random is not None, function is not None]
        if sum(not_none) > 1:
            msg = "Either params or random or function must be provided, not more than one"
            raise ValueError(msg)
        if params is None and random is None and function is None:
            # If nothing is provided, create a V function with random parameters
            self.params = VFunction.random_params(n)
        self.n = n
        self._create_V()

    def __call__(self, i) -> float:
        """Call the :math:`V` function."""
        return self.V(i)

    def _create_V(self) -> None:
        """Create the :math:`V` function."""
        n = self.n
        if self.params is not None:
            alpha, beta, gamma, i_0, displacement = self.params
            # Use numpy vectorized operations
            indices = np.arange(n)
            r = (np.abs(indices - i_0) ** gamma) * ((np.sin(beta * indices)) * (indices / n) ** alpha + 1)
            r = np.roll(r, displacement)

        elif self.random is not None:
            r = [random.random() for _ in range(n)]
        elif self.function is not None:
            r = [self.function(i) for i in range(n)]
        else:
            msg = "Either params or random or function must be provided in the VFunction class"
            raise ValueError(msg)

        V_u = lambda i: r[i]  # noqa
        max_value = max([V_u(i) for i in range(n)])
        V = (lambda i: V_u(i) / max_value)  # noqa

        self.V = V

    def plot(self, name=None, sizes=None, saving_folder=None, figures_path=None) -> None:
        """
        Plot the :math:`V` function.
        
        :param name: The name of the file to save the plot. Optional, default is None.
        :type name: str
        :param sizes: The sizes of the plot. Optional, default is None.
        :type sizes: tuple
        :param saving_folder: The folder to save the plot. Optional, default is None.
        :type saving_folder: str
        :param figures_path: The path to the figures folder. Optional, default is None.
        """
        n = self.n
        if sizes is not None:
            labelsize = sizes[0]
            title_size = sizes[1]
            ysize = sizes[2]
            xsize = sizes[3]
            legend_size = sizes[4]
            figsize = sizes[5]
        else:
            labelsize = 14
            title_size = 15
            ysize = 14
            xsize = 14
            legend_size = 14  # Unused here
            figsize = None
        latex_settings(plt, mathpazo=True, labelsize=labelsize, figsize=figsize)
        ylabel = r"$V$"
        xlabel = r"$x$"
        if self.params is not None:
            if self.params[-1] != 0:
                print('Delta is not 0')
                title = r"$\alpha={:.3f}\,, \beta={:.3f}\,, \gamma={:.3f}\,, x_0={}\,, \delta={}$".format(*self.params)
            else:
                title = r"$\alpha={:.3f}\,, \beta={:.3f}\,, \gamma={:.3f}\,, x_0={}$".format(*self.params[:-1])
        else:
            title = r"Random V"
        plt.title(title, fontsize=title_size)
        plt.plot([self.V(j) for j in range(n)], color="darkgray")
        plt.ylabel(ylabel, fontsize=ysize)
        plt.xlabel(xlabel, fontsize=xsize)
        if name is not None:
            if saving_folder is None:
                if figures_path is None:
                    # Define a figures directory relative to the current working directory and ensure it exists
                    figures_path = os.path.join(os.getcwd(), "figures")
                    os.makedirs(figures_path, exist_ok=True)
                saving_path = os.path.join(figures_path, name)
            else:
                saving_path = os.path.join(saving_folder, name)

            plt.savefig(saving_path)
        else:
            plt.show()
        plt.close()

    def return_index(self, case=None) -> tuple or str:
        if self.params is not None:
            params_formatted = (f"{self.params[0]:.3f}", f"{self.params[1]:.3f}", f"{self.params[2]:.3f}",
                                f"{self.params[3]}", f"{self.params[4]}")
            return params_formatted
        if self.random is not None:
            return f"Random case {case}"
        if self.function is not None:
            return f"Function case {case}"
        msg = "Either params or random or function must be provided in the VFunction class"
        raise ValueError(msg)

    @staticmethod
    def random_params(n) -> tuple:
        """Generate random parameters for the parametric form of the function :math:`V`."""
        alpha = random.random()
        # beta = random.random() / (n/100)
        beta = 2 * random.random()
        gamma = 2 * random.random()
        i_0 = random.randint(0, n)
        displacement = random.randint(0, n)

        return alpha, beta, gamma, i_0, displacement

    def best_phis(self, heuristics, no_elements, additional_best=None, delta_rho=None, best_heuristics=None) -> tuple:
        """Find the best :math:`\\phi` functions according to the expected value of the maximum of :math:`V` for a given set of heuristics.

        :param V: The function :math:`V` to be maximized
        :type V: function
        :param heuristics: The heuristics to be used for obtaining the best phi functions
        :type heuristics: list
        :param no_elements: The number of best heuristics to find
        :type no_elements: int
        :param additional_best: (optional, default is None) The additional best phi functions to 
            consider. If this is not None, the best heuristics are found first, then the additional 
            best phi functions are added to the best heuristics if their expected value is higher.
        :type additional_best: list of :class:`PhiFunction`
        :param best_heuristics: (optional, default is None) The best heuristics to be used. If this is 
            not None, the best heuristics are used directly.
        :type best_heuristics: list
        :param delta_rho: (optional, default is None) The delta_rho to be used. If None, the best heuristics are the heuristics. 
            If not None, the best heuristics are the radii of the best phi functions.
        :type delta_rho: list
        :return: The best phi functions, the best heuristics, the number of elements that come from 
            the additional best, the number of distinct heuristics in the best heuristics, and the 
            number of distinct expected values in the best heuristics
        :rtype: tuple
        """
        cache = {}

        # Use cached values if delta_rho is not None, as there are repeated calls
        def sorting_function_no_cache(h):
            if delta_rho is not None:
                return PhiFunction(V=self, radius=h).expected_value[0]
            else:
                return PhiFunction(V=self, heuristics=h).expected_value[0]

        def sorting_function_cache(h):
            h_tuple = tuple(h)
            if h_tuple in cache.keys():
                return cache[h_tuple]
            else:
                res = sorting_function_no_cache(h)
                cache[h_tuple] = res
                return res

        if delta_rho is not None:
            sorting_function = sorting_function_cache
        else:
            sorting_function = sorting_function_no_cache

        if best_heuristics is None:
            sorted_heur = sorted(heuristics, key=sorting_function, reverse=True)
            best_heuristics = sorted_heur[:no_elements]
        else:
            best_heuristics = best_heuristics
        if delta_rho is not None:
            best_phi = [PhiFunction(V=self, radius=h) for h in best_heuristics]
        else:
            best_phi = [PhiFunction(V=self, heuristics=h) for h in best_heuristics]

        if additional_best is not None:
            sorted_additional_best = sorted(additional_best, key=lambda p: p.expected_value[0], reverse=True)
            # Add additional best phi functions to the best ones if their expected value is higher. Record the number of
            # elements that come from the additional best. As it is sorted, we find the intersection
            count = 0
            for phi in sorted_additional_best:
                if phi.expected_value[0] > best_phi[-1].expected_value[0]:
                    best_phi[-1] = phi
                    best_phi = sorted(best_phi, key=lambda p: p.expected_value[0], reverse=True)
                    count += 1
                else:
                    break  # The list is sorted, so we can break when the expected value is lower
            intersection = count

        else:
            intersection = None

        # Find duplicates in the best heuristics
        if delta_rho is not None:
            # Best heuristics are the radii of the best phi functions
            best_heuristics = [phi.radius for phi in best_phi]
        sorted_best_heur = [sorted(h) for h in best_heuristics]
        distinct_heur_sorted = len(set(tuple(h) for h in sorted_best_heur))
        distinct_heur_exp = len(set([PhiFunction.expected_value_static(h=h, V=self)[0] for h in best_heuristics]))

        return best_phi, best_heuristics, intersection, distinct_heur_sorted, distinct_heur_exp

    def create_additional_bests(self, add_radial_search=False, additional_best_no=10, search=5,
                                exclude_max=True, delta_rho=None) -> list[PhiFunction]:
        """Create additional best :math:`\\phi` functions.

        :param add_radial_search: (optional, default is False) Flag to include best :math:`\\phi` functions that search the maximizer of :math:`V` in a region given by the radius centered at the starting point
        :type add_radial_search: bool
        :param additional_best_no: (optional, default is 10) The number of additional best :math:`\\phi` functions to create
        :type additional_best_no: int
        :param search: (optional, default is :math:`n//20`) The number of points to add to the search region where the maximizer of :math:`V` is searched. This is a subset of the points in the search region.
        :type search: int
        :param exclude_max: (optional, default is True) Flag to exclude the global maximizer from the search region, see :meth:`phi_region` for more details
        :type exclude_max: bool
        :param delta_rho: (optional, default is None) The delta_rho to be used. If None, 
            the search region is the pool of points. If not None, the search region is the pool of points plus the radius.
        :type delta_rho: list
        :return: The additional best :math:`\\phi` functions
        :rtype: list of :class:`PhiFunction`
        """
        phi_best = []
        V = self.V
        n = self.n
        if exclude_max:
            global_maximizer = [j for j in range(n) if V(j) == 1]
            pool = [j for j in range(n) if j not in global_maximizer]
        else:
            pool = range(n)
        for _ in range(additional_best_no):
            region = random.sample(pool, search)  # n//20 is another option
            phi = PhiFunction(V=self, region=region)
            phi_best.append(phi)
        if add_radial_search and delta_rho is not None:
            # Append the phi function that searches the maximizer of V in a region given by the radius
            pairs = PhiGroup.create_pool_tools(delta_rho_ratio=[delta_rho[0] + 1, delta_rho[1]], repeat=1)
            # Smaller delta_rho_ratio are considered in pool for random_group
            # pairs = create_pool_tools(delta_rho_ratio=[delta_rho[1]-1, delta_rho[1]], repeat=1)

            # Takes less time, but less functions
            # pairs = random.sample(pairs, 5)
            for pair in pairs:
                phi_rad = PhiFunction(V=self, radius=pair)
                phi_best.append(phi_rad)
        return phi_best
