# coding=utf-8
# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" MindSpore GPT-J model."""

from typing import Tuple, Union

import mindspore
import mindspore.numpy as np
import math
from mindspore import nn
from mindspore import ops
import mindspore.ops.operations as P
import mindspore.ops.functional as F

from .configuration_gptj import GPTJConfig


class GPTJAttention(nn.Cell):
    def __init__(self, config):
        super().__init__()

        self.max_positions = config.max_position_embeddings
        self.masked_bias = mindspore.Parameter(-1e9, requires_grad=False)

        self.attn_dropout = nn.Dropout(1 - config.attn_pdrop)
        self.resid_dropout = nn.Dropout(1 - config.resid_pdrop)

        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            return None
            # raise ValueError(
            #     f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and `num_attention_heads`: {self.num_attention_heads})."
            # )
        self.softmax = P.Softmax(axis=-1)
        self.cat1 = P.Concat(axis=-1)
        self.cat2 = P.Concat(axis=-2)
        self.transpose = P.Transpose()
        self.cast = P.Cast()
        self.sqrt = P.Sqrt()
        self.expand_dims = P.ExpandDims()
        self.pow = P.Pow()
        self.div = P.Div()
        self.stack = P.Stack(axis=-1)
        self.sin = P.Sin()
        self.cos = P.Cos()
        self.mul = P.Mul()
        self.add = P.Add()
        self.sub = P.Sub()
        self.matmul = P.MatMul()
        self.bmatmul = P.BatchMatMul()
        self.reshape = P.Reshape()
        # self.split = P.Split(3, 2)
        # self.slice = P.StridedSlice()

        self.scale_attn = self.sqrt(mindspore.Tensor(self.head_dim, dtype=mindspore.float32))

        self.k_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=False)
        self.v_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=False)
        self.q_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=False)
        self.out_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=False)
        self.rotary_dim = config.rotary_dim

        inv_freq = self.div(1.0, self.pow(10000, self.div(np.arange(0, self.rotary_dim, 2).astype(mindspore.float32), self.rotary_dim)))
        inv_freq = self.expand_dims(inv_freq, 0)
        seq = np.arange(self.max_positions).astype(mindspore.float32)
        seq = self.expand_dims(seq, 1)
        sinusoid_inp = self.matmul(seq, inv_freq)
        self.sins = self.sin(sinusoid_inp)
        self.coss = self.cos(sinusoid_inp)

    def _split_heads(self, tensor, num_attention_heads, attn_head_size, rotary):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        new_shape = F.shape(tensor)[:-1] + (num_attention_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        if rotary:
            return tensor
        if len(F.shape(tensor)) == 5:  # (batch, blocks, block_length, head, head_features)
            return self.transpose(tensor, (0, 1, 3, 2, 4))  # (batch, blocks, head, block_length, head_features)
        elif len(F.shape(tensor)) == 4:  # (batch, seq_length, head, head_features)
            return self.transpose(tensor, (0, 2, 1, 3))  # (batch, head, seq_length, head_features)
        else:
            return None
            # raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        if len(F.shape(tensor)) == 5:
            tensor = self.transpose(tensor, (0, 1, 3, 2, 4))
        elif len(F.shape(tensor)) == 4:
            tensor = self.transpose(tensor, (0, 2, 1, 3))
        else:
            return None
            # raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")
        new_shape = F.shape(tensor)[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _rotate_every_two(self, x):
        # x_shape = F.shape(x)
        # x1 = self.slice(x, (0, 0, 0, 0), x_shape, (1, 1, 1, 2))
        # x2 = self.slice(x, (0, 0, 0, 1), x_shape, (1, 1, 1, 2))
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x = self.stack((-x2, x1))
        x_shape = F.shape(x)
        shape = x_shape[:-2] + (x_shape[-2] * x_shape[-1],)
        return self.reshape(x, shape)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')

    def _apply_rotary_pos_emb(self, x, sins, coss, offset=0):
        sin = ops.repeat_elements(self.expand_dims(self.expand_dims(sins[:x.shape[1]], 1), 0), 2, 3)
        cos = ops.repeat_elements(self.expand_dims(self.expand_dims(coss[:x.shape[1]], 1), 0), 2, 3)
        return (x * cos) + (self._rotate_every_two(x) * sin)

    def _attn(
        self,
        query,
        key,
        value,
        causal_mask,
        attention_mask=None,
        head_mask=None,
    ):

        # compute causal mask from causal mask buffer

        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = self.cast(query, mindspore.float32)
        key = self.cast(key, mindspore.float32)

        attn_weights = self.bmatmul(query, key.transpose((0, 1, 3, 2)))
        inverse_mask = self.sub(
            self.cast(F.tuple_to_array((1.0,)), attn_weights.dtype),
            self.cast(causal_mask, attn_weights.dtype)
        )
        adder = self.mul(inverse_mask, attn_weights)
        attn_weights = self.mul(causal_mask, attn_weights)
        attn_weights = self.add(attn_weights, adder)

        attn_weights = attn_weights / self.scale_attn

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = self.softmax(attn_weights)
        attn_weights = self.cast(attn_weights, value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = self.bmatmul(attn_weights, value)

        return attn_output

    def construct(
        self,
        hidden_states,
        attention_bias,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
    ):

        query = self.q_proj(hidden_states)  # (batch, seq_length, hidden_dim)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_attention_heads, self.head_dim, True)  # (batch, seq_length, head, head_features)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, True)
        value = self._split_heads(value, self.num_attention_heads, self.head_dim, False)

        seq_len = F.shape(key)[1]
        offset = 0

        if layer_past is not None:
            offset = F.shape(layer_past[0])[-2]
            seq_len += offset

        if self.rotary_dim is not None:
            # key_shape = F.shape(key)
            # k_rot = self.slice(key, (0, 0, 0, 0), (key_shape[0], key_shape[1], key_shape[2], self.rotary_dim), (1, 1, 1, 1))
            # k_pass = self.slice(key, (0, 0, 0, self.rotary_dim), (key_shape[0], key_shape[1], key_shape[2], key_shape[3]), (1, 1, 1, 1))
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            # q_rot = self.slice(query, (0, 0, 0, 0), (key_shape[0], key_shape[1], key_shape[2], self.rotary_dim), (1, 1, 1, 1))
            # q_pass = self.slice(query, (0, 0, 0, self.rotary_dim), (key_shape[0], key_shape[1], key_shape[2], key_shape[3]), (1, 1, 1, 1))
            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            k_rot = self._apply_rotary_pos_emb(k_rot, self.sins, self.coss, offset=offset)
            q_rot = self._apply_rotary_pos_emb(q_rot, self.sins, self.coss, offset=offset)
        
            key = self.cat1((k_rot, k_pass))
            query = self.cat1((q_rot, q_pass))
        else:
            key = self._apply_rotary_pos_emb(key, self.sins, self.coss, offset=offset)
            query = self._apply_rotary_pos_emb(query, self.sins, self.coss, offset=offset)

        key = self.transpose(key, (0, 2, 1, 3))
        query = self.transpose(query, (0, 2, 1, 3))

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = self.cat2((past_key, key))
            value = self.cat2((past_value, value))

        # compute self-attention: V x Softmax(QK^T)
        query_length, key_length = query.shape[-2], key.shape[-2]
        # causal_mask = attention_bias
        causal_mask = attention_bias[:, :, key_length - query_length : key_length, :key_length]
        attn_output = self._attn(query, key, value, causal_mask, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output


class GPTJMLP(nn.Cell):
    def __init__(self, intermediate_size, config):  # in MLP: intermediate_size= 4 * embed_dim
        super().__init__()
        embed_dim = config.n_embd

        self.fc_in = nn.Dense(embed_dim, intermediate_size)
        self.fc_out = nn.Dense(intermediate_size, embed_dim)

        self.dropout = nn.Dropout(1 - config.resid_pdrop)

        self.tanh = P.Tanh()
        self.pow = P.Pow()
        self.gelu_const = math.sqrt(2.0 / math.pi)

    def gelu_new(self, x):
        return 0.5 * x * (1.0 + self.tanh(self.gelu_const * (x + 0.044715 * self.pow(x, 3.0))))

    def construct(self, hidden_states):
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.gelu_new(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPTJBlock(nn.Cell):
    def __init__(self, config):
        super().__init__()
        inner_dim = 4 * config.n_embd
        self.ln_1 = nn.LayerNorm((config.n_embd,), epsilon=config.layer_norm_epsilon)
        print("Begin constructing GPTJAttention...")
        self.attn = GPTJAttention(config)
        self.mlp = GPTJMLP(inner_dim, config)

    def construct(
        self,
        hidden_states,
        attention_bias,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(
            hidden_states,
            attention_bias,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

        return hidden_states


class GPTJPreTrainedModel(nn.Cell):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)


class GPTJModel(GPTJPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        print("Embedding...")
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.drop = nn.Dropout(1 - config.embd_pdrop)
        print("Begin CellList...")
        self.h = nn.CellList([GPTJBlock(config) for _ in range(config.n_layer)])
        print("CellList complete")
        self.ln_f = nn.LayerNorm((self.embed_dim,), epsilon=config.layer_norm_epsilon)
        self.tril = nn.Tril()
        self.ones = P.Ones()
        self.expand_dims = P.ExpandDims()
        self.cast = P.Cast()
        print("Begin attention bias...")
        self.attention_bias = mindspore.Parameter(self.tril(self.ones((config.max_position_embeddings, config.max_position_embeddings), mindspore.uint8)).view(
                1, 1, config.max_position_embeddings, config.max_position_embeddings
            ), requires_grad = False)
        print("Attention bias complete!")

        self.position_ids = np.arange(config.max_position_embeddings, dtype=mindspore.int64)

        # Initialize weights and apply final processing
        # self.post_init()
    
    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def get_head_mask(self, head_mask, num_hidden_layers: int, is_attention_chunked: bool = False):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = self.expand_dims(head_mask, -1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        if head_mask.dim() == 1:
            head_mask = self.expand_dims(self.expand_dims(self.expand_dims(self.expand_dims(head_mask, 0), 0), -1), -1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = self.expand_dims(self.expand_dims(self.expand_dims(head_mask, 1), -1), -1)  # We can specify head_mask for each layer
        # assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = self.cast(head_mask, self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def construct(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            return None
            # raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            return None
            # raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = (None,) * len(self.h)
        else:
            past_length = past_key_values[0][0].shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:input_shape[-1]]
            position_ids = self.expand_dims(position_ids, 0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            # assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = self.expand_dims(self.expand_dims(attention_mask, 1), 1)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = self.cast(attention_mask, self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_attention_heads x N x N
        # head_mask has shape n_layer x batch x num_attention_heads x N x N
        head_mask = self.get_head_mask(head_mask, len(self.h))

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.shape[-1],)

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            hidden_states = block(
                hidden_states,
                attention_bias=self.attention_bias,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
            )

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)

        return hidden_states


class GPTJForCausalLM(GPTJPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.cast = P.Cast()
        print("Begin constructing GPTJModel...")
        self.transformer = GPTJModel(config)
        print("GPTJModel constructed.")
        self.lm_head = nn.Dense(config.n_embd, config.vocab_size)

        # Initialize weights and apply final processing
        # self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def construct(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        transformer_output = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        lm_logits = self.cast(self.lm_head(transformer_output), mindspore.float32)

        return lm_logits
