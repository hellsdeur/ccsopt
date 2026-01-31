import time
import tracemalloc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

class Stats:
    def __init__(self, n_runs: int, algorithm, *args, **kwargs):
        self.n_runs = n_runs
        self.algorithm = algorithm
        self.args = args
        self.kwargs = kwargs
        self.data = {
            "iteration": [],
            "best_solution": [],
            "best_value": [],
            "history": [],
            "execution_time": [],
            "memory_peak": [],
            "memory_peak_mb": [],
        }

    def _run_single(self, run_index):
        start_time = time.time()
        tracemalloc.start()
        instance = self.algorithm(*self.args, **self.kwargs)
        global_best_position, global_best_value, history = instance.run()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.time()

        return {
            "iteration": run_index,
            "best_solution": global_best_position,
            "best_value": global_best_value,
            "history": history,
            "execution_time": end_time - start_time,
            "memory_peak": peak,
            "memory_peak_mb": peak / 10**6,
        }

    def run(self):
        with Pool(cpu_count()) as pool:
            results = list(tqdm(pool.imap(self._run_single, range(self.n_runs)), total=self.n_runs, desc=f"Running {self.algorithm.__name__}", unit="run"))

        for entry in results:
            for key in self.data:
                self.data[key].append(entry[key])

    def frame(self):
        return pd.DataFrame(self.data)

    def plot_convergence(self, title: str, savefig: str = None):
        df = self.frame()

        value_by_run = np.array(df["history"].tolist())
        value_by_iteration = value_by_run.T

        means_by_iteration = np.mean(value_by_iteration, axis=1)
        std_by_iteration = np.std(value_by_iteration, axis=1)
        max_value = np.nanmax(value_by_iteration)

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.lineplot(
            x=np.arange(len(means_by_iteration)),
            y=means_by_iteration,
            ax=ax,
            label="Média do valor",
            color="royalblue",
            linewidth=2.5
        )

        ax.fill_between(
            np.arange(len(means_by_iteration)),
            means_by_iteration - std_by_iteration,
            means_by_iteration + std_by_iteration,
            color="royalblue",
            alpha=0.2,
            label="Desvio padrão do valor"
        )

        ax.axhline(
            y=max_value,
            linestyle="--",
            color="forestgreen",
            linewidth=2,
            label=f"Valor máximo = {max_value}"
        )

        ax.set_xlabel("Iteração", fontsize=16)
        ax.set_ylabel("Valor", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
        ax.legend(fontsize=12, frameon=True)
        plt.tight_layout()

        if savefig:
            plt.savefig(savefig, dpi=600)

        return fig, ax