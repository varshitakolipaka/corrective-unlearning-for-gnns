import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

method_to_name = {
    "cacdc": "Cognac",
    "yaum": "AC/DC",
    "gif": "GIF",
    "gnndelete": "GNNDelete",
    "retrain": "Retrain",
    "scrub": "SCRUB",
    "utu": "UtU",
    "megu": "MEGU",
}

method_to_color = {
    "Retrain": "grey",
    "AC/DC": "#4C72B0",
    "Cognac": "#729ECE",
    "GNNDelete": "#D95F02",
    "MEGU": "#7570B3",
    "GIF": "#E7298A",
    "UtU": "#66A61E",
    "SCRUB": "#E6AB02",
}


# Function to plot the time taken by each method from a CSV file
def plot_time_taken(csv_file):
    # Read CSV file into DataFrame
    df = pd.read_csv(csv_file)

    sns.set_style("whitegrid")

    # Extract and convert time_taken to numeric values, ignoring uncertainty
    df["time_taken"] = pd.to_numeric(
        df["time_taken"].str.split("±").str[0], errors="coerce"
    )

    # Filter out rows with missing or NaN time_taken values
    df = df.dropna(subset=["time_taken"])

    # change method names to human readable names, and if not found, delete the row
    df["Method"] = df["Method"].apply(lambda x: method_to_name.get(x, np.nan))
    
    # remove acdc from the list
    df = df[df["Method"] != "AC/DC"]

    # Plotting with numbers on top of the bars
    plt.figure(figsize=(6, 6))
    ax = sns.barplot(x="Method", y="time_taken", data=df, color='#DAE8FC')
    
    # add borders to bars
    for i, bar in enumerate(ax.patches):
        bar.set_edgecolor('black')
        bar.set_linewidth(1)
    
    
    # plt.xticks(rotation=45, ha='right')
    # plt.xlabel('Method')
    # plt.ylabel('Time Taken')
    plt.title("Cora", fontsize=20)

    # increase size of x-axis labels
    plt.xticks(fontsize=14, rotation=45, ha="right")
    plt.yticks(fontsize=12)
    
    # make the first x tick bold
    ax.get_xticklabels()[0].set_fontweight('bold')

    plt.ylabel("Time Taken (s)", fontsize=14)
    plt.xlabel("")

    # Add numbers on top of the bars
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        

        
        ax.annotate(
            f"{height:.3f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold" if i == 0 else "normal",
        )

    plt.tight_layout()
    plt.savefig("time_taken.svg")


# Example usage
plot_time_taken("logs/label_cf_logs/Cora/run_logs_label_0.5_5_63_cf_0.75.csv")
