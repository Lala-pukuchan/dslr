import matplotlib.pyplot as plt
import seaborn as sns


def pair_plot(data, columns, hue):
    """
    plot a pair plot
    """
    grid = sns.PairGrid(data, vars=columns, hue=hue, height=1)
    grid.map_lower(sns.scatterplot)
    grid.add_legend()
    plt.savefig("./visualizations/scatter_plots/pair_plot.png")   
    plt.close()
