import os
os.environ["TRANSFORMERS_DISABLE_ONNX"] = "1"
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import (
    BertAttention,
    BertIntermediate, 
    BertOutput,
    BertPreTrainedModel
)
from transformers import PretrainedConfig

TransformerLayerNorm = torch.nn.LayerNorm

class MyConfig(PretrainedConfig):

    def __init__(
        self,
        k=5,
        max_hop_dis_index=100,
        max_inti_pos_index=100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=32,
        hidden_act="gelu",
        hidden_dropout_prob=0.5,
        attention_probs_dropout_prob=0.3,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_decoder=False,
        batch_size=64,
        window_size=1,
        weight_decay=5e-4,
        **kwargs
    ):
        super(MyConfig, self).__init__(**kwargs)
        self.max_hop_dis_index = max_hop_dis_index
        self.max_inti_pos_index = max_inti_pos_index
        self.k = k
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.is_decoder = is_decoder
        self.batch_size = batch_size
        self.window_size = window_size
        self.weight_decay = weight_decay

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.config = config
        
        # Initialize layers with proper weight initialization
        self.layer = nn.ModuleList([
            TransformerLayer(config) 
            for _ in range(config.num_hidden_layers)
        ])
        
        self.register_buffer(
            'attention_mask_template',
            torch.ones((config.batch_size, config.max_inti_pos_index))
        )

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None):
        
        # Initialize containers for outputs
        all_hidden_states = () if self.output_hidden_states else None
        all_attentions = () if self.output_attentions else None
        
        # Process through layers
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i] if head_mask is not None else None,
                encoder_hidden_states,
                encoder_attention_mask
            )
            
            hidden_states = layer_outputs[0]
            
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Final hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Prepare outputs
        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
            
        return outputs

class EdgeEncoding(nn.Module):
    def __init__(self, config):
        super(EdgeEncoding, self).__init__()
        self.config = config
        
        # Initialize embeddings
        self.inti_pos_embeddings = nn.Embedding(
            config.max_inti_pos_index,
            config.hidden_size,
            padding_idx=0
        )
        self.hop_dis_embeddings = nn.Embedding(
            config.max_hop_dis_index,
            config.hidden_size,
            padding_idx=0
        )
        self.time_dis_embeddings = nn.Embedding(
            config.max_hop_dis_index,
            config.hidden_size,
            padding_idx=0
        )
        
        # Initialize normalization and dropout
        self.input_dropout = nn.Dropout(0.2)
        self.LayerNorm = TransformerLayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0,
                std=self.config.initializer_range
            )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, init_pos_ids=None, hop_dis_ids=None, time_dis_ids=None):
        # Input validation and conversion
        if not all(x is not None for x in [init_pos_ids, hop_dis_ids, time_dis_ids]):
            raise ValueError("All input IDs must be provided")
            
        device = init_pos_ids.device
        
        # Convert to long dtype
        init_pos_ids = init_pos_ids.long()
        hop_dis_ids = hop_dis_ids.long()
        time_dis_ids = time_dis_ids.long()
        
        # Validate input shapes
        if not (init_pos_ids.shape == hop_dis_ids.shape == time_dis_ids.shape):
            raise ValueError("All input shapes must match")
            
        # Generate embeddings
        position_embeddings = self.inti_pos_embeddings(init_pos_ids)
        hop_embeddings = self.hop_dis_embeddings(hop_dis_ids)
        time_embeddings = self.time_dis_embeddings(time_dis_ids)
        
        # Combine embeddings
        embeddings = position_embeddings + hop_embeddings + time_embeddings
        
        # Apply normalization and dropout
        embeddings = self.input_dropout(embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None):
        
        # Self attention
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        # Cross attention for decoder
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]

        # Feed-forward network
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        
        return (layer_output,) + outputs