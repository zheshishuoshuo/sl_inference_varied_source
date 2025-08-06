import numpy as np
import pandas as pd
import os

def generate_dummy_A_phys_table(muDM_grid, sigmaDM_grid, betaDM_grid, xiDM_grid, filename):
    rows = []
    for mu in muDM_grid:
        for sigma in sigmaDM_grid:
            for beta in betaDM_grid:
                for xi in xiDM_grid:
                    rows.append({
                        'mu_DM': mu,
                        'sigma_DM': sigma,
                        'beta_DM': beta,
                        'xi_DM': xi,
                        'A_phys': 1.0  # 统一设为 1
                    })
    filename = os.path.join(os.path.dirname(__file__), '..', 'tables', filename)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"[INFO] Dummy A(η) 表格已写入：{filename}")

# === 例子：使用和主程序一致的网格 ===

if __name__ == "__main__":
    muDM_grid    = np.linspace(12, 13, 100)
    sigmaDM_grid = np.linspace(0.1, 0.5, 100)
    betaDM_grid  = np.linspace(1.0, 3.0, 100)
    xiDM_grid    = [0]

    generate_dummy_A_phys_table(
        muDM_grid, sigmaDM_grid, betaDM_grid, xiDM_grid,
        filename="A_phys_table_4D_dummy.csv"
    )
