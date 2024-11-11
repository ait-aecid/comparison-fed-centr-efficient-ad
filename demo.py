from dataloader import load_data


args = {
    "dataset_path": "datasets/BGL",
    "amount_clients": 3,
    "seed_number": 2,
    "train_per": 0.1,
}


for i in range(args["amount_clients"]):
    print(f"Client {i}")
    for run in range(4):
        print(f"Run {run}")
        print(data := load_data(config=args, num_client=i, num_run=run))

       #print(data.train[:3])

