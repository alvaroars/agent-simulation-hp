import os
import sys

# Add the parent directory containing the code
code_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, code_path)

# Add the simulations directory itself
simulations_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, simulations_path)


# -- Project information -----------------------------------------------------
project = 'Agent-based simulations'
author = 'Álvaro Romaniega'
release = '1.0'
master_doc = "index"

# -- Extensions configuration ------------------------------------------------
extensions = [
    # Documentation generation
    "sphinx.ext.autodoc",
    "numpydoc",  # needs to be loaded *after* autodoc
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    
    # Visualization
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    
    # Cross-referencing
    "sphinx.ext.intersphinx",
    
    # Additional features
    "sphinx_toolbox.installation",
    "sphinx_toolbox.latex",
    "sphinx_design",
    "nbsphinx",
]

# PDF embedding configuration
pdfembed_embed_options = {
    "width": "100%",
    "height": "800px",
}

# -- Numpydoc configuration -------------------------------------------------
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_listbool = True
numpydoc_xref_param_typebool = True

# -- Autodoc configuration --------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'private-members': True,
    'inherited-members': False,
    'show-inheritance': True,
}
autodoc_typehints = "none"
autodoc_member_order = "bysource"

# -- Autosummary and sections -----------------------------------------------
autosummary_generate = True
autosectionlabel_prefix_document = True

# -- HTML output configuration ----------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "collapse_navigation": False,
}

html_show_sourcelink = True
python_module_index = False

html_sidebars = {
    "**": [
        "globaltoc.html",
        "searchbox.html",
    ]
}

# Templates and static files
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Files to exclude from documentation
exclude_patterns = []

# -- Diagram output configuration -------------------------------------------
graphviz_output_format = "svg"
inheritance_node_attrs = dict(
    shape="ellipse", fontsize=12, height=0.75, color="dodgerblue1", style="filled"
)
inheritance_graph_attrs = dict(fontsize=12, size='"16.0, 20.0"')
#graphviz_dot = 'C:\Program Files\Graphviz\\bin\dot.exe'  # Uncomment for Windows

# -- LaTeX output configuration ---------------------------------------------
latex_engine = "pdflatex"
latex_elements = {
    "preamble": r"""
\usepackage[titles]{tocloft}
\cftsetpnumwidth {1.25cm}\cftsetrmarg{1.5cm}
\setlength{\cftchapnumwidth}{0.75cm}
\setlength{\cftsecindent}{\cftchapnumwidth}
\setlength{\cftsecnumwidth}{1.25cm}
""",
    "fncychap": r"\usepackage[Bjornstrup]{fncychap}",
    "printindex": r"\footnotesize\raggedright\printindex",
}
latex_show_urls = "footnote"
latex_documents = [(master_doc, "main.tex", project, author, "report")]
