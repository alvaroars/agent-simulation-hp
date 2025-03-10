import os

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
results_path = os.path.join(os.path.dirname(__file__), 'results')
tables_path = os.path.join(repo_path, "tables")
figures_path = os.path.join(repo_path, "figures")


if __name__ == "__main__":
    print("Configuration paths:")
    print("results_path:", results_path)
    print("tables_path:", tables_path)
    print("figures_path:", figures_path)
    print("repo_path:", repo_path)