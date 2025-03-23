Agent-based simulations in computational philosophy and social choice theory
============================================================================

This documentation covers a comprehensive simulation framework for studying agent-based models in the Hong-Page context, a widely used framework in the literature of collective intelligence, which lies in the intersection of the fields of social choice theory, economics, epistemology, and political philosophy. The library's modular structure provides a solid foundation for exploring and extending the original model. Researchers can easily implement alternative agent behaviors, problem landscapes, and group dynamics while leveraging the core infrastructure for visualization, analysis, and result processing. This extensible architecture enables systematic investigation of model variations and refinements to advance our understanding of collective problem-solving.

.. note::
   For a complete theoretical background and research findings, please refer to:
   
   - `[Rom23] <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4505907>`_ *Fatal Errors and Misuse of Mathematics in the Hong-Page Theorem and Landemore's Epistemic Argument*,

   - `[Rom25] <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5172653>`_ *Mathematical Analysis of the Hong-Page Framework*,

   - `Less technical summary: <https://alvaroromaniega.wordpress.com/2023/10/22/misuse-of-mathematics-in-the-hong-page-theorem-and-landemores-epistemic-argument-for-democracy/>`_ *Misuse of Mathematics in the Hong-Page Theorem and Landemore's Epistemic Argument for Democracy*,

   - My previous research on the area, `[Rom22] <https://www.sciencedirect.com/science/article/pii/S0165489622000543>`_ *On the Probability of the Condorcet Jury Theorem or the Miracle of Aggregation*, Mᴀᴛʜᴇᴍᴀᴛɪᴄᴀʟ Sᴏᴄɪᴀʟ Sᴄɪᴇɴᴄᴇs, Elsevier, Volume 119, September 2022, Pages 41-55.
   
   Specifically, the library implements the improvements presented in Section 6 of the second paper, and this documentation follows the notation presented there.

Overview       
--------

This simulation framework implements models related to the Hong-Page framework, including the improvements presented in `[Rom25] <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5172653>`_. The Hong-Page framework is a mathematical model of collective problem-solving that explores how agent diversity and cognitive abilities affect group performance. It gained prominence for its claim that "diversity trumps ability" - that under certain conditions, diverse groups can outperform groups composed of individually higher-performing agents. The framework includes both theoretical results (the Hong-Page Theorem) and agent-based simulations.

The library provides a flexible architecture for exploring how agent diversity, cognitive abilities, and collaboration patterns influence collective intelligence and problem-solving effectiveness in the aforementioned framework. Functions for incorporating results directly into LaTeX code are available, as well as tools for exporting to Excel and generating various plots to visualize simulation outcomes.

Key Features
-------------

- **Core simulation components**: The framework includes `simulator_dynamics` for creating value functions, agent heuristics, and group behaviors. Its modular structure accommodates various agent types, value functions, and group dynamics.
- **Agent diversity and abilities**: The system supports modeling agents with different cognitive capabilities through various ability coefficients and heuristic combinations, allowing for research on diversity effects in problem-solving.
- **Group dynamics analysis**: The framework provides tools for studying group problem-solving processes, including relay dynamics, disagreement cycles, and collaborative decision-making patterns.
- **Performance assessment**: Users can evaluate expected performance and efficiency across different agent configurations and group compositions.
- **LaTeX integration**: The automatic creation of LaTeX tables and captions for results.
- **Excel compatibility**: Import/export simulation results to Excel for broader accessibility and collaborative analysis. 
- **Visualization tools**: automatic elaboration of plots from the simulations and excel results is possible. LaTeX equations can be integrated into the plots.



.. toctree::
   :maxdepth: 4
   :caption: Contents:

   simulator_dynamics
   functions
   test_simulator
   config

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Getting Started
---------------

Toy example
...........

1. Define a value function (``VFunction``) representing the problem landscape
2. Create agent heuristics using ``PhiFunction``
3. Form groups using ``PhiGroup``
4. Run simulations and analyze results

.. code-block::    python

    from simulator_dynamics import VFunction, PhiFunction, PhiGroup
    
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
    expected_value, expected_iterations = group.expected_value()
    print(f"Expected solution quality: {expected_value}")
    print(f"Expected number of iterations: {expected_iterations}")

For more examples, see the :doc:`test_simulator` module.
