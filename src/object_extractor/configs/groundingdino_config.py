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
pre_norm = True
enforce_input_proj = False

# Transformer
hidden_dim = 256
pad_token_id = 0
max_position_embeddings = 256
query_dim = 4
num_feature_levels = 4
nheads_fusion = 8
transformer_activation = "relu"  # Added
dec_n_points = 4  # Added
enc_n_points = 4  # Added
two_stage_num_proposals = 900  # Added
assign_first_stage = True  # Added
with_mask = True  # Added
mask_dim = 256  # Added
freeze_text_encoder = True  # Added
random_refpoints_xy = False  # Added

# Positional Encoding
pe_temperatureH = 20
pe_temperatureW = 20
pe_temperature = 20

# Backbone Config
backbone = 'swin_T_224_1k'
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
use_checkpoint = True

# Model Parameters
dilation = False
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