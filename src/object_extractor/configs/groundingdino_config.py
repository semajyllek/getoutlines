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
transformer_activation = "relu"
dec_n_points = 4
enc_n_points = 4

# Text Cross Attention Parameters
use_text_cross_attention = True  # Added
text_dropout = 0.1  # Added
fusion_header_type = "text"  # Added

# Transformer Checkpoint Parameters
use_transformer_ckpt = True
use_checkpoint_for_decoder = True
use_checkpoint_for_encoder = True

# Fusion Parameters
use_fusion_layer = True
use_checkpoint_for_fusion = True
fusion_dropout = 0.1
fusion_droppath = 0.1

# Two-Stage Parameters
two_stage_type = 'standard'
two_stage_pat_embed = 0
two_stage_add_query_num = 0
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
two_stage_learn_wh = False
two_stage_default_hw = 0.1
two_stage_keep_all_tokens = False
two_stage_num_proposals = 900
assign_first_stage = True
embed_init_tgt = True
use_text_enhancer = False

# Additional Model Parameters
with_mask = True
mask_dim = 256
freeze_text_encoder = True
random_refpoints_xy = False
dn_labelbook_size = 100

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