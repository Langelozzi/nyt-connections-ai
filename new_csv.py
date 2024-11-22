import random

import pandas as pd

# Read the CSV file
csv = pd.read_csv("Connections_Data.csv")

# Initialize lists to hold the results
full_words = []
group_words = []

# Group words by Game ID and then by Group Name
for game_id, game_group in csv.groupby("Game ID"):
    game_words = []  # Temporary list to hold all words for this Game ID
    group_list = []  # Temporary list to hold grouped words by Group Name

    for group_name, group in game_group.groupby("Group Name"):
        words = group["Word"].tolist()  # List of words for this group
        group_list.append(words)  # Add the grouped words to group_list
        game_words.extend(words)  # Add words to the game-wide list

    # Append results for this Game ID
    for i in range(4):
        ##Remove if you dont want random puzzles in csv
        random.shuffle(game_words)
        full_words.append(game_words)  # All words for the Game ID
    for group in group_list:
        group_words.append(group)  # Grouped words for the Game ID

# Output example
print("Words grouped by Game ID:", full_words[:1])  # Display first Game ID
print("Groups within Game ID:", group_words[:1])  # Display groups of first Game ID

dataframe = pd.DataFrame({
    "Puzzle": full_words,
    "Answer": group_words
}
)

dataframe["Puzzle"] = dataframe["Puzzle"].apply(lambda x: ";".join(map(str, x)))
dataframe["Answer"] = dataframe["Answer"].apply(lambda x: ";".join(map(str, x)))


dataframe.to_csv("Connection_Answers.csv", index=False)

