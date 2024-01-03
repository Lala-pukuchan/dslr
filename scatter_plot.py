import matplotlib.pyplot as plt


def scatter_plot(numeric_data):
    """
    plot a scatter plot
    """
    # Calculate correlation coefficient
    corr_matrix = numeric_data.corr()
    print(corr_matrix)

    # Storage for max correlation coefficient
    max_corr = 0
    max_pair_feature = (None, None)

    # Find the pair of features with max correlation coefficient
    for i in range(corr_matrix.shape[0]):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > max_corr:
                max_corr = abs(corr_matrix.iloc[i, j])
                max_pair_feature = (corr_matrix.index[i], corr_matrix.columns[j])

    # print the max correlation coefficient and the pair of features
    print(f"Max correlation coefficient: {max_corr}")
    print(f"Max pair of features: {max_pair_feature}")

    # Plot the scatter plot
    plt.close()
    plt.scatter(numeric_data[max_pair_feature[0]], numeric_data[max_pair_feature[1]])
    plt.xlabel(max_pair_feature[0])
    plt.ylabel(max_pair_feature[1])
    plt.title(f"Scatter plot of {max_pair_feature[0]} and {max_pair_feature[1]}")
    plt.savefig("./visualizations/scatter_plot.png")
    plt.close()
