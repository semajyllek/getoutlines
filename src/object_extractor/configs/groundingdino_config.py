# src/object_extractor/configs/groundingdino_config.py

modelname = "groundingdino"
batch_size_train = 16
batch_size_eval = 8
lr_backbone = 1e-5
lr = 1e-4
max_text_len = 256
text_encoder_type = "bert-base-uncased"

# GroundingDINO Architecture
num_queries = 900
position_embedding = "sine"
enc_layers = 6
dec_layers = 6
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_patterns = 0
text_threshold = 0.25
box_threshold = 0.35

# Positional Encoding
pe_temperatureH = 20
pe_temperatureW = 20
pe_temperature = 20

# Model Parameters
use_checkpoint = True
checkpoint_activations = True
aux_loss = True
with_box_refine = True
two_stage = True

# Random
seed = 42
distributed = False

# Train
num_workers = 4
device = 'cuda'
world_size = 1
eval_skip = 1
output_dir = "outputs/run1"
resume = False
start_epoch = 0
num_epochs = 50

# Dataset
train_dataset_name = "mixed"
test_dataset_name = "mixed"