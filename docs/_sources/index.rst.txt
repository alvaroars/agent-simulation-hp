.. simulations-HP documentation master file, created by sphinx-quickstart.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Agent-based simulations
==================================
This documentation covers the simulation framework for studying agent-based simulations. 

.. note::
   For a complete theoretical background and research findings, please refer to:
   
   - `[Rom23] <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4505907>`_



Contents
--------

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   simulator_dynamics
   functions
   test_simulator
   config

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Overview
========

This simulation framework implements models related to the Hong-Page diversity model. It provides tools for:

* Creating and evaluating value functions (V functions)
* Implementing agent heuristics and problem-solving strategies (Phi functions)
* Analyzing group dynamics and collective problem-solving
* Testing and comparing different configurations of agents and problems

The framework is designed to be flexible and extensible, allowing for a variety of experimental setups and analysis approaches.

Getting Started
===============

To use this framework, you typically:

1. Define a value function (VFunction) representing the problem landscape
2. Create agent heuristics using PhiFunction
3. Form groups using PhiGroup
4. Run simulations and analyze results

.. code-block::    python

    from simulations.simulator_dynamics import VFunction, PhiFunction, PhiGroup
    
    # Create a value function
    n = 100  # Problem space size
    V = VFunction(n=n, random=True)
    
    # Create agent heuristics
    agent1 = PhiFunction(V=V, heuristics=[1, 2, 3])
    agent2 = PhiFunction(V=V, heuristics=[2, 4, 6])
    
    # Form a group
    group = PhiGroup(phi_list=[agent1, agent2])
    
    # Run group dynamics
    starting_point = 0
    solution, iterations = group(starting_point)
    
    # Calculate expected performance
    expected_value, expected_iterations = group.expected_value_group()
    print(f"Expected solution quality: {expected_value}")
    print(f"Expected number of iterations: {expected_iterations}")

