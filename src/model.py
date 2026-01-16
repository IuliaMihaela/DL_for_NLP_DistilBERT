import torch
import torch.nn as nn
from transformers import BertModel, DistilBertConfig, DistilBertForMaskedLM

class DistilBertStudent(nn.Module):
    def __init__(self, teacher_model_name="bert-base-uncased"):
        super().__init__()
        # Load teacher configuration to match the hidden size of 768
        self.teacher = BertModel.from_pretrained(teacher_model_name)
        
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
        print("Start initialization...")
        
        # Copy Embeddings
        self.student.distilbert.embeddings.word_embeddings.weight.data = \
            self.teacher.embeddings.word_embeddings.weight.data.clone()
        self.student.distilbert.embeddings.position_embeddings.weight.data = \
            self.teacher.embeddings.position_embeddings.weight.data.clone()
        self.student.distilbert.embeddings.LayerNorm.load_state_dict(
            self.teacher.embeddings.LayerNorm.state_dict()
        )

        # Copy Transformer Layers
        for i in range(6):
            teacher_layer_idx = i * 2
            
            # Access the specific layer objects
            std_layer = self.student.distilbert.transformer.layer[i]
            tch_layer = self.teacher.encoder.layer[teacher_layer_idx]
            
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

        print(f"Successfully initialized student with layers: {[i*2 for i in range(6)]}")

    def forward(self, input_ids, attention_mask=None):
        outputs = self.student(
            input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True
        )
        return outputs.logits, outputs.hidden_states