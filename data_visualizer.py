import matplotlib.pyplot as plt


def draw_pie_chart(data):
    # print(data.CATEGORY.value_counts())
    spam_count = data.CATEGORY.value_counts()[1]
    ham_count = data.CATEGORY.value_counts()[0]
    labels = ['Spam', 'Ham']
    sizes = [spam_count, ham_count]
    custom_colors = ['#c23616', '#487eb0']
    # offset = [0.05, 0.05]
    # labels = ['Spam', 'Ham', 'Lamb', 'Cam']
    # sizes = [30, 40, 20, 10]
    # custom_colors = ['#c23616', '#487eb0', '#e1b12c', '#4cd137']
    # offset = [0.05, 0.05, 0.05, 0.05]

    plt.figure(figsize=[3, 3], dpi=254)
    plt.pie(
        sizes,
        labels=labels,
        textprops={'fontsize': 9},
        startangle=90,
        autopct='%1.0f%%',
        colors=custom_colors,
        # explode=offset,
        pctdistance=0.8
    )

    # plt.show()

    # Donut Chart
    centre_circle = plt.Circle((0, 0), radius=0.6, fc='white')
    plt.gca().add_artist(centre_circle)

    plt.show()

