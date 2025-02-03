from typing import List, Optional, Tuple
import torch
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertPooler
from codes.Component import EdgeEncoding, TransformerEncoder


BertLayerNorm = torch.nn.LayerNorm

class BaseModel(BertPreTrainedModel):
    data = None

    def __init__(self, config):
        super(BaseModel, self).__init__(config)
        self.config = config

        self.embeddings = EdgeEncoding(config)
        self.encoder = TransformerEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.raw_feature_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.raw_feature_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def setting_preparation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[Optional[torch.Tensor]]]:
        """
        Prepare model settings with proper validation
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Initialize attention mask
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        
        # Initialize token type IDs
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # Process attention mask
        extended_attention_mask: torch.Tensor
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(torch.long)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Process encoder attention mask if in decoder mode
        encoder_extended_attention_mask = None
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            
            encoder_extended_attention_mask = self._prepare_encoder_attention_mask(
                encoder_attention_mask, encoder_hidden_shape, device
            )

        # Process head mask
        head_mask = self._prepare_head_mask(head_mask, self.config.num_hidden_layers, device)

        return token_type_ids, extended_attention_mask, encoder_extended_attention_mask, head_mask


    def forward(
        self,
        init_pos_ids: torch.Tensor,
        hop_dis_ids: torch.Tensor,
        time_dis_ids: torch.Tensor,
        head_mask: Optional[List[Optional[torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with input validation
        """
        device = init_pos_ids.device
        
        # Validate inputs
        if not all(isinstance(x, torch.Tensor) for x in [init_pos_ids, hop_dis_ids, time_dis_ids]):
            raise TypeError("All inputs must be torch.Tensor")
            
        if not (init_pos_ids.device == hop_dis_ids.device == time_dis_ids.device):
            raise ValueError("All inputs must be on same device")
        
        if head_mask is None:
            head_mask = [None] * self.config.num_hidden_layers

        # Forward computation
        embedding_output = self.embeddings(
            init_pos_ids=init_pos_ids,
            hop_dis_ids=hop_dis_ids, 
            time_dis_ids=time_dis_ids
        )
        encoder_outputs = self.encoder(embedding_output, head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        return (sequence_output, pooled_output,) + encoder_outputs[1:]

    def _prepare_encoder_attention_mask(
        self, 
        encoder_attention_mask: torch.Tensor,
        encoder_hidden_shape: Tuple[int, int],
        device: torch.device
    ) -> torch.Tensor:
        """Prepare encoder attention mask"""
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for encoder_attention_mask (shape {encoder_attention_mask.shape})")
            
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)
        return (1.0 - encoder_extended_attention_mask) * -10000.0

    def _prepare_head_mask(
        self,
        head_mask: Optional[torch.Tensor],
        num_hidden_layers: int,
        device: torch.device
    ) -> List[Optional[torch.Tensor]]:
        """Prepare head mask"""
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=self.dtype)
        return [None] * num_hidden_layers if head_mask is None else head_mask


    def run(self):
        pass