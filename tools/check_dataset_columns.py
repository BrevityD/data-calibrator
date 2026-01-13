from datacalibrator.datasets.code_adaptor import get_code_dataset

train_ds, _ = get_code_dataset(size=1)
print(train_ds[0].keys())
print("Entry point:", train_ds[0].get("entry_point"))
print("Test sample:", train_ds[0].get("test")[:100])
