import matplotlib.pyplot as plt
# from eda_file import reading_files
def visualize_eda():

    df_cust.customer_state.value_counts().plot(kind='pie', figsize=(8, 10), autopct='%.1f%%', radius=2)
    plt.legend()
    plt.show()
    # Top 10 cities with their value counts
    df_cust.customer_city.value_counts().sort_values(ascending=False)[:10]

