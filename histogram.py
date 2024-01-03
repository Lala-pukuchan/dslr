import matplotlib.pyplot as plt


def histogram(data):
    """
    Create a histogram
    """
    # List of courses
    courses = [
        "Arithmancy",
        "Astronomy",
        "Herbology",
        "Defense Against the Dark Arts",
        "Divination",
        "Muggle Studies",
        "Ancient Runes",
        "History of Magic",
        "Transfiguration",
        "Potions",
        "Care of Magical Creatures",
        "Charms",
        "Flying",
    ]

    # Create a histogram for each course
    for cource in courses:
        plt.figure(figsize=(10, 6))
        for house in data["Hogwarts House"].unique():
            plt.hist(
                data[data["Hogwarts House"] == house][cource].dropna(),
                bins=20,
                alpha=0.5,
                label=house,
            )
        plt.title(f"{cource} distribution")
        plt.xlabel("Grades")
        plt.ylabel("Number of students")
        plt.legend()
        plt.savefig(f"./visualizations/histograms/{cource}.png")
        plt.close()
