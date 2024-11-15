
from dataloader import load_data
from tqdm import tqdm

def read_files(path):
    data = []
    with open(path, "r") as f:
        for row in f.readlines():
            data.append(row.replace("\n", "").split(",")[-1])
            data[-1] = [int(d) for d in data[-1].split(" ") if d != ""]
    return data


path_abnormal = "/home/garcia-gomeza/Documents/DB/HDFS_max/hdfs_test_abnormal"
path_normal = "/home/garcia-gomeza/Documents/DB/HDFS_max/hdfs_test_normal"
path_train = "/home/garcia-gomeza/Documents/DB/HDFS_max/hdfs_train"


args = {
    "dataset_path": "datasets/HDFS",
    "amount_clients": 1,
    "seed_number": 2,
    "train_per": 0.1,
}


data = load_data(args, num_client=0, num_run=0)
print(data)

test_abnormal = read_files(path_abnormal)
test_normal = read_files(path_normal)
test_normal.extend(read_files(path_train))

print("Test Abnormal")
j = 0
print(len(test_abnormal), len(data.test_abnormal))
for row in tqdm(data.test_abnormal):
    if row not in test_abnormal:
        print(row)
        j += 1
print(f"{'OK' if j == 0 else 'FAIL'}")

print("Test normal")
j = 0
print(len(test_normal), len(data.test_normal))
for row in tqdm(data.test_normal):
    if row not in test_normal:
        print(row)
        j += 1
print(f"{'OK' if j == 0 else 'FAIL'}")

print("Train")
j = 0
print(len(test_normal), len(data.train))
for row in tqdm(data.train):
    if row not in test_normal:
        print(row)
        j += 1
print(f"{'OK' if j == 0 else 'FAIL'}")