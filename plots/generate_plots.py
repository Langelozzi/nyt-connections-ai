import matplotlib.pyplot as plt

def epochs_vs_win_percentage(epochs: list[int]):
    win_percentages = [0.083, 0.083, 0.083, 0.083, 0.089, 0.095, 0.089, 0.089, 0.089, 0.083]

    plt.figure(figsize=(8,5))
    plt.plot(
        epochs,
        win_percentages,
        marker='o',
        linestyle='-',
        color='y',
        label='Win Percentage'
    )
    plt.title('Win Percentage vs. Number of Epochs')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Win Ratio')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Annotate specific points (optional)
    for x, y in zip(epochs, win_percentages):
        plt.text(x, y, f'{y:.3f}', fontsize=9, ha='center')

    # Show or save the plot
    plt.tight_layout()
    plt.savefig('epochs_vs_win_percentage.png', dpi=300)


def epochs_vs_total_connections_ratio(epochs: list[int]):
    connection_ratio = [0.202, 0.202, 0.204, 0.204, 0.212, 0.215, 0.21, 0.21, 0.21, 0.21]

    plt.figure(figsize=(8,5))
    plt.plot(
        epochs,
        connection_ratio,
        marker='o',
        linestyle='-',
        color='y',
        label='Connections ratio'
    )
    plt.title('Connection Ratio vs. Number of Epochs')
    plt.xlabel('Connection Ratio')
    plt.ylabel('Win Ratio')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Annotate specific points (optional)
    for x, y in zip(epochs, connection_ratio):
        plt.text(x, y, f'{y:.3f}', fontsize=9, ha='center')

    # Show or save the plot
    plt.tight_layout()
    plt.savefig('epochs_vs_connection_ratio.png', dpi=300)

def main():
    epochs = [1, 3, 5, 7, 9, 10, 11, 12, 15, 20]
    epochs_vs_win_percentage(epochs)
    epochs_vs_total_connections_ratio(epochs)


if __name__ == '__main__':
    main()