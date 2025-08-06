import os
import numpy as np
from .compute_norm_grid import build_A_phys_table_parallel_4D


def main():
    """Compute the A(eta) grid using default parameters."""
    muDM_grid = np.linspace(12, 13, 100)
    sigmaDM_grid = np.linspace(0.1, 0.5, 100)
    betaDM_grid = np.linspace(1.0, 3.0, 100)
    xiDM_grid = [0]

    table_path = os.path.join(os.path.dirname(__file__), '..', 'tables', 'A_phys_table_4D.csv')

    build_A_phys_table_parallel_4D(
        muDM_grid,
        sigmaDM_grid,
        betaDM_grid,
        xiDM_grid,
        n_samples=2000,
        filename=table_path,
    )


if __name__ == "__main__":
    main()
