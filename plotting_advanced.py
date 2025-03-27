import argparse
import json
import os
import pprint
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots


def get_frac_size_from_filename(filename):
    """
    Extracts the fraction size from the filename.
    """
    if "_cf_" in filename:
        return float(filename.split("_cf_")[-1].replace(".json", ""))
    else:
        return 1.0


def clean_data(seed_data):
    """
    Cleans the data by removing all other metrics except forget

    Args:
        seed_data: The data to clean

    Returns:
        The cleaned data
    """

    # new dict to store the cleaned data
    cleaned_data = {}

    for method in seed_data.keys():
        for metric in seed_data[method].keys():
            if metric == "forget":
                cleaned_data[method] = seed_data[method][metric]

    return cleaned_data


def load_data(input_dir, attack_type):
    data_total = {}
    for f in os.listdir(input_dir):
        if f.endswith(".json"):

            if attack_type not in f:
                continue
                
            try:
                data = json.load(open(os.path.join(input_dir, f)))
            except Exception as e:
                print(f"An error occurred while loading {f}: {e}")
                # try to load as a string and convert to dict
                with open(os.path.join(input_dir, f)) as file:
                    data_str = file.read()
                    print(data_str)
                
            print(data.keys())
            frac_size = get_frac_size_from_filename(f)

            data_total[frac_size] = data["results"]
            # remove average and std_dev keys
            
            try:
                del data_total[frac_size]["average"]
                del data_total[frac_size]["standard_dev"]
            except KeyError:
                pass
            
            # clean the data
            for seed in data_total[frac_size].keys():
                data_total[frac_size][seed] = clean_data(data_total[frac_size][seed])

    return data_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot forget scores across different fraction sizes."
    )
    parser.add_argument("--i", type=str, help="Folder containing the JSON files.")
    parser.add_argument(
        "--a",
        type=str,
        help="Attack type to plot",
        default="label",
        choices=["label", "edge"],
    )

    args = parser.parse_args()

    data = load_data(args.i, args.a)
    pprint.pprint(data)

    """
    Data structure now looks like this:
     0.75: {'0': {'cacdc': 0.7601835915088927,
              'contra_2': 0.7316121629374641,
              'gif': 0.49386115892139987,
              'gnndelete': 0.0,
              'megu': 0.5133677567412507,
              'original': 0.7279977051061388,
              'poisoned': 0.5164658634538153,
              'retrain': 0.5970166379804934,
              'scrub': 0.28519793459552495,
              'utu': 0.20298336201950662,
              'yaum': 0.5674698795180724},
        '1': {'cacdc': 0.6220883534136545,
              'contra_2': 0.7316121629374641,
              'gif': 0.49862306368330467,
              'gnndelete': 0.0,
              'megu': 0.48284566838783705,
              'original': 0.7279977051061388,
              'poisoned': 0.5164658634538153,
              'retrain': 0.6188755020080321,
              'scrub': 0.28519793459552495,
              'utu': 0.20298336201950662,
              'yaum': 0.5674698795180724},
        '2': {'cacdc': 0.6220883534136545,
              'contra_2': 0.7316121629374641,
              'gif': 0.49036144578313257,
              'gnndelete': 0.0,
              'megu': 0.5469879518072289,
              'original': 0.7279977051061388,
              'poisoned': 0.5164658634538153,
              'retrain': 0.621399885255307,
              'scrub': 0.28519793459552495,
              'utu': 0.20298336201950662,
              'yaum': 0.5674698795180724},
        '3': {'cacdc': 0.6220883534136545,
              'contra_2': 0.7316121629374641,
              'gif': 0.4951233505450373,
              'gnndelete': 0.0,
              'megu': 0.5378657487091222,
              'original': 0.7279977051061388,
              'poisoned': 0.5164658634538153,
              'retrain': 0.5503729202524383,
              'scrub': 0.28519793459552495,
              'utu': 0.20298336201950662,
              'yaum': 0.5674698795180724},
        '4': {'cacdc': 0.7601835915088927,
              'contra_2': 0.7316121629374641,
              'gif': 0.4951233505450373,
              'gnndelete': 0.0,
              'megu': 0.5545611015490534,
              'original': 0.7279977051061388,
              'poisoned': 0.5164658634538153,
              'retrain': 0.5890418818129661,
              'scrub': 0.28519793459552495,
              'utu': 0.20298336201950662,
              'yaum': 0.5674698795180724}},
     1.0: {'0': {'cacdc': 0.6308663224325874,
             'contra_2': 0.6932300631095811,
             'gif': 0.4182444061962134,
             'gnndelete': 0.4857142857142857,
             'megu': 0.5193918531267929,
             'original': 0.7279977051061388,
             'poisoned': 0.5164658634538153,
             'retrain': 0.6369477911646586,
             'scrub': 0.22748135398737807,
             'utu': 0.1620768789443488,
             'yaum': 0.49391853126792884},
       '1': {'cacdc': 0.6308663224325874,
             'contra_2': 0.6932300631095811,
             'gif': 0.427768215720023,
             'gnndelete': 0.5,
             'megu': 0.4768215720022949,
             'original': 0.7279977051061388,
             'poisoned': 0.5164658634538153,
             'retrain': 0.4706827309236947,
             'scrub': 0.22748135398737807,
             'utu': 0.1620768789443488,
             'yaum': 0.49391853126792884},
       '2': {'cacdc': 0.6308663224325874,
             'contra_2': 0.6932300631095811,
             'gif': 0.4182444061962134,
             'gnndelete': 0.5,
             'megu': 0.5473895582329317,
             'original': 0.7279977051061388,
             'poisoned': 0.5164658634538153,
             'retrain': 0.4992541594951233,
             'scrub': 0.22748135398737807,
             'utu': 0.1620768789443488,
             'yaum': 0.49391853126792884},
       '3': {'cacdc': 0.6308663224325874,
             'contra_2': 0.6932300631095811,
             'gif': 0.427768215720023,
             'gnndelete': 0.5,
             'megu': 0.5336775674125072,
             'original': 0.7279977051061388,
             'poisoned': 0.5164658634538153,
             'retrain': 0.49994262765347103,
             'scrub': 0.22748135398737807,
             'utu': 0.1620768789443488,
             'yaum': 0.49391853126792884},
       '4': {'cacdc': 0.6308663224325874,
             'contra_2': 0.6932300631095811,
             'gif': 0.4182444061962134,
             'gnndelete': 0.4523809523809524,
             'megu': 0.5450372920252439,
             'original': 0.7279977051061388,
             'poisoned': 0.5164658634538153,
             'retrain': 0.6002294893861159,
             'scrub': 0.22748135398737807,
             'utu': 0.1620768789443488,
             'yaum': 0.49391853126792884}}}
             
    We have to convert this to a pandas dataframe with the following columns:
    - frac_size
    - method
    - seed
    - forget
    """
    
    data_list = []
    for frac_size in data.keys():
        for seed in data[frac_size].keys():
            for method in data[frac_size][seed].keys():
                data_list.append({
                    "frac_size": frac_size,
                    "method": method,
                    "seed": seed,
                    "forget": data[frac_size][seed][method]
                })
                
    df = pd.DataFrame(data_list)
    
    print(df.head())
    
    # now, plot the data using seaborn
    
    # x axis is frac_size, y axis is forget, hue is method, grouped by seed to get aggregate forget scores and error around the mean
    
    plt.style.use(['science', 'no-latex', 'grid'])
    plt.figure(figsize=(4, 4))
    
    #  add a column to indicate if the method is a baseline
    baselines = ['utu', 'scrub', 'megu', 'gif', 'gnndelete', 'yaum', 'contra_2', 'finetune']
    
    df['is_baseline'] = df['method'].apply(lambda x: x in baselines)
    
    # sort methods to ensure consistent color assignment
    df['method'] = pd.Categorical(df['method'], categories=sorted(df['method'].unique()))
    
    # color by method
    # colors = plt.cm.Dark1(range(len(df['method'].unique())))
    
    # give markers to each method
    markers = ['o', 's', 'v', '^', '>', '<', 'D', 'P', 'X', 'H', 'd', 'p', 'x', 'h']
    
    # limit y axis to highest and lowest forget scores
    upper_limit = df['forget'].max()
    
    # lower limit is the poison forget score
    lower_limit = df[df['method'] == 'poisoned']['forget'].min()
    
    plt.ylim(lower_limit - 0.1, upper_limit + 0.05)
    
    for method in df['method'].unique():
        # plot the average forget score for each method
        df_method = df[df['method'] == method]
        print(df_method)
        # drop all cols except frac_size and forget
        df_method = df_method[['frac_size', 'forget']]
        
        # group by frac_size and calculate the mean and std dev
        df_method = df_method.groupby('frac_size').agg(['mean', 'std']).reset_index()
        print(df_method)
        
        # if baseline, make the plot transparent
        
        # plot the data
        plt.plot(df_method['frac_size'], df_method['forget']['mean'], label=method, marker=markers.pop(), ls='-' if method not in baselines else '--')
    
    # ensure ticks are displayed at all frac sizes
    plt.xticks(df['frac_size'].unique())
    
    # make the plot more readable
    plt.xlabel("Fraction Size")
    plt.ylabel("Forget Score")
    
    # add legend outside the plot below
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    # ensure the plot is not cut off
    plt.tight_layout()
    
    # save the plot in the same folder as the input data
    plt.savefig(os.path.join(args.i, f'forget_scores_{args.a}.png'))