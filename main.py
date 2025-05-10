# Import project-specific modules
from data import load_dataset
from log import log_to_json
from config import CONFIG




# -----------------------------Test Case: log_to_json-----------------------------

# print(f"\n🔧 Batch size from config: {CONFIG.BATCH_SIZE}")

# train_data, train_labels, test_data, test_labels = load_dataset()

# log_to_json(key="train_overview", record={
#     "samples": len(train_data),
#     "shape": list(train_data[0].shape),
#     "labels": len(set(train_labels))
# })

# log_to_json(key="test_overview", record={
#     "samples": len(test_data),
#     "shape": list(test_data[0].shape),
#     "labels": len(set(test_labels))
# })

# print("\n✅ main.py successfully executed")

# -------------------------------------------------------------------------------
