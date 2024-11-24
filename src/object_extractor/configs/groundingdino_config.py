# src/object_extractor/configs/groundingdino_config.py

modelname = "groundingdino"

# Model Architecture
num_queries = 900
hidden_dim = 256
position_embedding = "sine"
text_encoder_type = "bert-base-uncased"
enc_layers = 6
dec_layers = 6
pre_norm = False
normalize_before = False

# Transformer Parameters
nheads = 8
dim_feedforward = 2048
dropout = 0.0
enc_n_points = 4
dec_n_points = 4
transformer_activation = "relu"
num_patterns = 0
num_feature_levels = 4

# Position Encoding
pe_temperatureH = 20
pe_temperatureW = 20
pe_temperature = 20
query_dim = 4
random_refpoints_xy = False

# Backbone Config
backbone = 'swin_T_224_1k'
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
use_checkpoint = True
dilation = False

# Detection Settings
text_threshold = 0.25
box_threshold = 0.35
with_box_refine = True
two_stage = True
two_stage_type = 'standard'
two_stage_num_proposals = 900
assign_first_stage = True
embed_init_tgt = True
use_text_enhancer = False
dec_pred_bbox_embed_share = True
dec_pred_class_embed_share = True
use_text_cross_attention = True
text_dropout = 0.1
use_fusion_layer = True
use_checkpoint_for_fusion = True
use_transformer_ckpt = True
use_checkpoint_for_encoder = True
use_checkpoint_for_decoder = True
sub_sentence_present = True
enforce_input_proj = False
return_intermediate_dec = True