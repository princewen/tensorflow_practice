import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def team_stats():
    team_stats_df = pd.read_csv('data/nba1.txt',sep=',')
    team_opp_stats_df = pd.read_csv('data/nba2.txt',sep=',')

    team_opp_stats_df.columns = ['Rk','Team','G'] + ['opp' + x for x in team_opp_stats_df.columns[3:]]

    mergedf = pd.merge(team_stats_df,team_opp_stats_df,on=['Team'])

    pts = np.array(mergedf[['Team','PTS','oppPTS']].values.tolist())

    index = np.arange(pts.shape[0])

    plt.bar(index,pts[:,1])
    plt.show()


def main():
    team_stats()

if __name__ == '__main__':
    main()