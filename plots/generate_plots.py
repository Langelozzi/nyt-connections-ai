import matplotlib.pyplot as plt
import numpy as np


def epochs_vs_win_percentage(epochs: list[int]):
    win_percentages = [0.083, 0.083, 0.083, 0.083, 0.089, 0.095, 0.089, 0.089, 0.089, 0.083]

    plt.figure(figsize=(10,6))
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


def epochs_vs_total_connections_accuracy(epochs: list[int]):
    connection_accuracy = [0.202, 0.202, 0.204, 0.204, 0.212, 0.215, 0.21, 0.21, 0.21, 0.21]

    plt.figure(figsize=(10,6))
    plt.plot(
        epochs,
        connection_accuracy,
        marker='o',
        linestyle='-',
        color='y',
        label='Connections accuracy'
    )
    plt.title('Connection Ratio vs. Number of Epochs')
    plt.xlabel('Connection Ratio')
    plt.ylabel('Win Ratio')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Annotate specific points (optional)
    for x, y in zip(epochs, connection_accuracy):
        plt.text(x, y, f'{y:.3f}', fontsize=9, ha='center')

    # Show or save the plot
    plt.tight_layout()
    plt.savefig('epochs_vs_connection_accuracy.png', dpi=300)


def compare_models(model_results: list[dict]):
    categories = [data['model_name'] for data in model_results]
    win_percentages = [data['win_percentage'] for data in model_results]
    connection_accuracies = [data['connection_accuracy'] for data in model_results]

    x = np.arange(len(categories))  # X-axis positions for categories
    width = 0.35  # Width of each bar

    # Create the plot
    plt.figure(figsize=(25, 10))
    plt.bar(x - width / 2, win_percentages, width, label='Win Percentage', color='blue', alpha=0.7)
    plt.bar(x + width / 2, connection_accuracies, width, label='Connection Accuracy', color='orange', alpha=0.7)

    # Add labels, title, and legend
    plt.xlabel('Models')
    plt.ylabel('Performance Ratio')
    plt.title('Win Percentage and Connection Accuracy of Different Models')
    plt.xticks(x, categories)
    plt.legend()

    # Add grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Display the graph
    # plt.tight_layout()
    plt.savefig('model_comparisons.png')


def main():
    # Comparing different influence of training data on existing word embeddings
        # epochs = [1, 3, 5, 7, 9, 10, 11, 12, 15, 20]
        # epochs_vs_win_percentage(epochs)
        # epochs_vs_total_connections_accuracy(epochs)

    # Comparing different models
    model_performance = [
        {
            'model_name': 'Google Word Embedding 3M',
            'win_percentage': 0.098,
            'connection_accuracy': 0.221
        },
        {
            'model_name': 'Google Word Embedding 1M',
            'win_percentage': 0.092,
            'connection_accuracy': 0.213
        },
        {
            'model_name': 'Google Word Embedding 500k',
            'win_percentage': 0.082,
            'connection_accuracy': 0.186
        },
        {
            'model_name': 'Continuous Skip Gram Embedding',
            'win_percentage': 0.066,
            'connection_accuracy': 0.182
        },
        {
            'model_name': 'Twitter Word2Vec Embeddings',
            'win_percentage': 0.068,
            'connection_accuracy': 0.203
        },
        {
            'model_name': 'Google Word Embedding 1M + 10 Epoch Training',
            'win_percentage': 0.095,
            'connection_accuracy': 0.215
        },
        {
            'model_name': 'CNN: 3M Google Word Embedding Trained on Archive',
            'win_percentage': 0.00,
            'connection_accuracy': 0.02,
        },
        {
            'model_name': 'Agglomerative Clustering (cosine, single) + Pre-Trained BERT',
            'win_percentage': 0.1311,
            'connection_accuracy': 0.401
        },
        {
            'model_name': 'Agglomerative Clustering (cosine, single) + Google Word Embedding 3M',
            'win_percentage': 0.1858,
            'connection_accuracy': 0.4057
        },
        {
            'model_name': 'Agglomerative Clustering (cosine, single) + Google Word Embedding 3M + Phrase Cleaning',
            'win_percentage': 0.2203,
            'connection_accuracy': 0.4641
        },
        {
            'model_name': 'Fuzzy Means Clustering + Google Word Embedding 3M + Phrase Cleaning',
            'win_percentage': 0.1686,
            'connection_accuracy': 0.3113
        },
        {
            'model_name': 'Medoid Clustering + Google Word Embedding 3M + Phrase Cleaning',
            'win_percentage': 0.0709,
            'connection_accuracy': 0.2146
        },
    ]
    compare_models(model_performance)


if __name__ == '__main__':
    main()