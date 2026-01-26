import torch
import torch.nn as nn
from transformers import DistilBertConfig, DistilBertForMaskedLM, BertForMaskedLM

class DistilBertStudent(nn.Module):
    def __init__(self, teacher_model_name="bert-base-uncased"):
        super().__init__()
        # Load teacher and freeze
        self.teacher = BertForMaskedLM.from_pretrained(teacher_model_name)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        # Student config
        self.config = DistilBertConfig(
            vocab_size=self.teacher.config.vocab_size,
            dim=self.teacher.config.hidden_size,
            n_layers=6, 
            n_heads=self.teacher.config.num_attention_heads,
            hidden_dim=4 * self.teacher.config.hidden_size,
        )
        self.student = DistilBertForMaskedLM(self.config)

    def initialize_from_teacher(self):
        """ Implements the 'one layer out of two' initialization """
        teacher_bert = self.teacher.bert
        print("Initializing student weights from teacher...")
        
        # Copy Embeddings
        self.student.distilbert.embeddings.word_embeddings.weight.data = teacher_bert.embeddings.word_embeddings.weight.data.clone()
        self.student.distilbert.embeddings.position_embeddings.weight.data = teacher_bert.embeddings.position_embeddings.weight.data.clone()
        self.student.distilbert.embeddings.LayerNorm.load_state_dict(teacher_bert.embeddings.LayerNorm.state_dict())

        # Copy every 2nd Transformer Layer
        for i in range(6):
            std_layer = self.student.distilbert.transformer.layer[i]
            tch_layer = teacher_bert.encoder.layer[i * 2]
            
            # Attention Mapping
            std_layer.attention.q_lin.weight.data = tch_layer.attention.self.query.weight.data.clone()
            std_layer.attention.q_lin.bias.data   = tch_layer.attention.self.query.bias.data.clone()
            std_layer.attention.k_lin.weight.data = tch_layer.attention.self.key.weight.data.clone()
            std_layer.attention.k_lin.bias.data   = tch_layer.attention.self.key.bias.data.clone()
            std_layer.attention.v_lin.weight.data = tch_layer.attention.self.value.weight.data.clone()
            std_layer.attention.v_lin.bias.data   = tch_layer.attention.self.value.bias.data.clone()
            std_layer.attention.out_lin.weight.data = tch_layer.attention.output.dense.weight.data.clone()
            std_layer.attention.out_lin.bias.data   = tch_layer.attention.output.dense.bias.data.clone()
            std_layer.sa_layer_norm.load_state_dict(tch_layer.attention.output.LayerNorm.state_dict())

            # FFN Mapping
            std_layer.ffn.lin1.weight.data = tch_layer.intermediate.dense.weight.data.clone()
            std_layer.ffn.lin1.bias.data   = tch_layer.intermediate.dense.bias.data.clone()
            std_layer.ffn.lin2.weight.data = tch_layer.output.dense.weight.data.clone()
            std_layer.ffn.lin2.bias.data   = tch_layer.output.dense.bias.data.clone()
            std_layer.output_layer_norm.load_state_dict(tch_layer.output.LayerNorm.state_dict())

    @torch.no_grad()
    def forward_teacher(self, input_ids, attention_mask=None):
        out = self.teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True, # Critical for Attention Distillation
            return_dict=True
        )
        return out.logits, out.hidden_states, out.attentions

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True, # Critical for Attention Distillation
            return_dict=True
        )
        return outputs.loss, outputs.logits, outputs.hidden_states, outputs.attentions