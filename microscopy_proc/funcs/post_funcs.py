import matplotlib.pyplot as plt
import seaborn as sns


def make_maxima_scatter(df):
    fig, ax = plt.subplots(figsize=(5, 10))
    sns.scatterplot(x=df["x"], y=df["y"], marker=".", alpha=0.2, s=10, ax=ax)
    ax.invert_yaxis()
