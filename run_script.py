import os

training_epochs=[1, 50, 100, 200, 300, 500, 600, 700, 1208]
for epoch in training_epochs:
    command = f"python main.py --db_name rebuttal --unlearning_model cacdc --training_epochs {epoch}"
    os.system(command)