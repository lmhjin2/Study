# Since we can't directly use PlantUML here, let's provide a Python code snippet that generates a textual representation
# of the BERT model architecture, which users can then use as a basis for creating their own diagrams or further exploration.

def print_bert_architecture():
    print("BERT Model Architecture Overview:")
    print("\n[Transformer Encoder]")
    print(" - Self-Attention Mechanism: Processes each word in the context of all other words in the sentence.")
    print(" - Feed Forward Network: Activates after each self-attention mechanism in each layer.")
    
    print("\n[Pre-Training Tasks]")
    print(" - Masked Language Model (MLM): Predicts randomly masked words in a sentence.")
    print(" - Next Sentence Prediction (NSP): Predicts if two sentences logically follow each other.")
    
    print("\n[Fine-Tuning]")
    print(" - Task-Specific Input/Output Adjustments: Adjusts the model for specific NLP tasks (e.g., question answering, sentiment analysis).")
    print(" - Additional Layers (optional): Adds task-specific layers during fine-tuning if necessary.")

# Display the architecture
print_bert_architecture()