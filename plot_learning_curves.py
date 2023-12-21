import numpy as np
import pickle
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pickle
sys.path.append("..")
sns.set()
from pathlib import Path

from common.plotting_utils import get_scores, generate_plot, get_all_run_titles, get_all_runs_and_subruns


def make_graphs(experiment_name,
                subdir,
                run_titles=None,
                smoothen=False,
                min_length=-1,
                only_longest=False,
                cumulative=False,
                all_seeds=False):

    if run_titles is None:
        print("Using all runs in experiment")
        run_titles = get_all_run_titles(experiment_name=experiment_name)

    run_titles = list(filter(lambda x: not "!" in x, run_titles))
    
        
    run_titles.sort()
    log_dirs = [
        os.path.join(experiment_name, run_title) for run_title in run_titles
    ]
    score_arrays = []
    good_run_titles = []
    for log_dir, run_title in zip(log_dirs, run_titles):
        try:
            scores = get_scores(log_dir,
                                subdir=subdir,
                                only_longest=only_longest,
                                min_length=min_length,
                                cumulative=cumulative)
            
            print("scores length: ", len(scores[0]))
            print("last score: ", np.mean(scores[0][-10:]))
            print("Run title", run_title)

            score_arrays.append(scores)
            good_run_titles.append(run_title)
            
            if all_seeds:
                for i, score in enumerate(scores):
                    score_arrays.append(np.array([score]))
                    good_run_titles.append(run_title + f"_{i+1}")

        except Exception as e:
            print(f"skipping {log_dir} due to error {e}")
            pass
    
    [
        generate_plot(score_array, run_title, smoothen=smoothen)
        for score_array, run_title in zip(score_arrays, good_run_titles)
    ]
    
    # plt.xlim(0, 500)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.ylabel(subdir.replace("_", " "),size=12)
    plt.xlabel("Iteration", size=12)
    plt.legend(loc=2,prop={"size": 11}, )
    plt.title(experiment_name, size=15)
    plt.savefig("./myImagePDF.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def main():
    """
    Change these options and directories to suit your needs
    """
    ## Defaults
    subdir = "evaluation_reward"
    smoothen = True
    min_length = -1
    only_longest = False
    cumulative = False
    all_seeds = False 


    experiment_name = "Walker_Factorized_TD3" # put the result folder here
    run_titles = get_all_run_titles(experiment_name)
    make_graphs(experiment_name,
                subdir,
                run_titles=run_titles,
                smoothen=smoothen,
                min_length=min_length,
                only_longest=only_longest,
                cumulative=cumulative,
                all_seeds=all_seeds)
        
if __name__ == '__main__':
   main()
