import torch
import torch.nn as nn
from transformers import DistilBertConfig, DistilBertForMaskedLM, BertForMaskedLM

class DistilBertStudent(nn.Module):
    def __init__(self, teacher_model_name="bert-base-uncased"):
        super().__init__()
        # Load teacher configuration to match the hidden size of 768
        #self.teacher = BertModel.from_pretrained(teacher_model_name)
        self.teacher = BertForMaskedLM.from_pretrained(teacher_model_name)

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        # Create Student config with 6 instead of 12 layers
        self.config = DistilBertConfig(
            vocab_size=self.teacher.config.vocab_size,
            dim=self.teacher.config.hidden_size,
            n_layers=6, 
            n_heads=self.teacher.config.num_attention_heads,
            hidden_dim=4 * self.teacher.config.hidden_size,
        )
        
        # Initialize the Student model
        self.student = DistilBertForMaskedLM(self.config)

    def initialize_from_teacher(self):
        """
        Implements the 'one layer out of two' initialization
        """
        teacher_bert = self.teacher.bert
        print("Start initialization...")
        
        # Copy Embeddings
        self.student.distilbert.embeddings.word_embeddings.weight.data = \
            teacher_bert.embeddings.word_embeddings.weight.data.clone()
        self.student.distilbert.embeddings.position_embeddings.weight.data = \
            teacher_bert.embeddings.position_embeddings.weight.data.clone()
        self.student.distilbert.embeddings.LayerNorm.load_state_dict(
            teacher_bert.embeddings.LayerNorm.state_dict()
        )

        # Copy Transformer Layers
        for i in range(6):
            teacher_layer_idx = i * 2
            
            # Access the specific layer objects
            std_layer = self.student.distilbert.transformer.layer[i]
            tch_layer = teacher_bert.encoder.layer[teacher_layer_idx]
            
            # Map Attention Weights (Query, Key, Value)
            std_layer.attention.q_lin.weight.data = tch_layer.attention.self.query.weight.data.clone()
            std_layer.attention.q_lin.bias.data   = tch_layer.attention.self.query.bias.data.clone()
            
            std_layer.attention.k_lin.weight.data = tch_layer.attention.self.key.weight.data.clone()
            std_layer.attention.k_lin.bias.data   = tch_layer.attention.self.key.bias.data.clone()
            
            std_layer.attention.v_lin.weight.data = tch_layer.attention.self.value.weight.data.clone()
            std_layer.attention.v_lin.bias.data   = tch_layer.attention.self.value.bias.data.clone()

            # Map Output Projector (copying weights/bias)
            std_layer.attention.out_lin.weight.data = tch_layer.attention.output.dense.weight.data.clone()
            std_layer.attention.out_lin.bias.data   = tch_layer.attention.output.dense.bias.data.clone()
            
            std_layer.sa_layer_norm.load_state_dict(
                tch_layer.attention.output.LayerNorm.state_dict()
            )

            # Map Feed Forward Network (FFN)
            std_layer.ffn.lin1.weight.data = tch_layer.intermediate.dense.weight.data.clone()
            std_layer.ffn.lin1.bias.data   = tch_layer.intermediate.dense.bias.data.clone()
            
            std_layer.ffn.lin2.weight.data = tch_layer.output.dense.weight.data.clone()
            std_layer.ffn.lin2.bias.data   = tch_layer.output.dense.bias.data.clone()
            
            std_layer.output_layer_norm.load_state_dict(
                tch_layer.output.LayerNorm.state_dict()
            )

        # Teacher MLM head parts
        t_dense = self.teacher.cls.predictions.transform.dense
        t_ln    = self.teacher.cls.predictions.transform.LayerNorm
        t_dec   = self.teacher.cls.predictions.decoder

        # Student MLM head parts (DistilBERT)
        s_dense = self.student.vocab_transform
        s_ln    = self.student.vocab_layer_norm
        s_proj  = self.student.vocab_projector

        # Copy transform dense
        s_dense.weight.data = t_dense.weight.data.clone()
        s_dense.bias.data   = t_dense.bias.data.clone()

        # Copy LayerNorm
        s_ln.load_state_dict(t_ln.state_dict())

        # Copy decoder/projection
        s_proj.weight.data = t_dec.weight.data.clone()

        # Copy bias if exists
        if s_proj.bias is not None and t_dec.bias is not None:
            s_proj.bias.data = t_dec.bias.data.clone()

        # Important: keep tied weights consistent
        self.student.tie_weights()

        print(f"Successfully initialized student with layers: {[i*2 for i in range(6)]}")
        print("MLM head copied from teacher.")

    @torch.no_grad()
    def forward_teacher(self, input_ids, attention_mask=None):
        """
        Teacher forward pass (frozen, no grad).
        Returns teacher logits and hidden states.
        """
        self.teacher.eval()
        out = self.teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        return out.logits, out.hidden_states


    def forward(self, input_ids, attention_mask=None, labels=None):
        #Student forward pass. 
        # If labels are provided, returns MLM loss too.
        outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        return outputs.loss, outputs.logits, outputs.hidden_states
