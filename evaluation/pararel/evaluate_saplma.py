from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import gather_object
import torch
import torch.nn as nn
import json
from tqdm.auto import tqdm
import random
from argparse import ArgumentParser
import os
import numpy as np
from eval import eval_acc, eval_ap

STOP = []
SURE = []
UNSURE = []

cache_dir = '/work/vita/nie/cache/huggingface/hub'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SAPLMAClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SAPLMAClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

def extract_features(model, tokenizer, text, layer_idx=16):
    # Register a hook to get the activations from the specified layer
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Attach the hook to the specified layer
    if hasattr(model, 'layers'):
        # For models like LLaMA
        handle = model.model.layers[layer_idx].register_forward_hook(get_activation(f'layer_{layer_idx}'))
    else:
        # For other models
        handle = list(model.modules())[layer_idx].register_forward_hook(get_activation(f'layer_{layer_idx}'))
    
    # Prepare input
    full_input = "Question:" + text + " Answer:"
    inputs = tokenizer(full_input, return_tensors="pt").to(model.device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Remove the hook
    handle.remove()
    
    # Get the activation and process it
    layer_output = activations[f'layer_{layer_idx}']
    
    # Average over sequence length to get a fixed-size representation
    # Convert bfloat16 to float32 before converting to numpy
    feature_vector = layer_output.mean(dim=1).squeeze().to(torch.float32).cpu().numpy()
    
    return feature_vector

def inference(input_text):

    full_input = f"Question: {input_text} Answer:"
    inputs = tokenizer(full_input,return_tensors="pt").to(0)
    ids = inputs['input_ids']
    outputs = model.generate(
                ids,
                max_new_tokens = 15,
                output_scores = True,
                return_dict_in_generate=True
            )
    logits = outputs['scores']
    output_sequence = []
    product = 1
    count = 0
    for i in logits:        #greedy decoding and calculate the confidence
        pt = torch.softmax(torch.Tensor(i[0]),dim=0)
        max_loc = torch.argmax(pt)
        
        if max_loc in STOP:
            break
        else:
            output_sequence.append(max_loc)  
            product *= torch.max(pt)
            count += 1
            
    if output_sequence:
        output_text = tokenizer.decode(output_sequence)
    else:
        output_text = ""

    return output_text, full_input

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='openlm-research/open_llama_3b')
    parser.add_argument('--domain', type=str, default="ID")
    parser.add_argument('--layer_idx', type=int, default=16, help='Layer index to extract features from')
    parser.add_argument('--classifier_path', type=str, default='saplma_results/saplma_classifier_open_llama_3b.pt')
    parser.add_argument('--seed', type=int, default=999, help='random seed')
    
    args = parser.parse_args()
    model_name = args.model.split('/')[-1]
    set_seed(args.seed)
    
    accelerator = Accelerator()
    
    # Load model and tokenizer
    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, unk_token="<unk>", bos_token="<s>", eos_token="</s>", add_bos_token=False, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', cache_dir=cache_dir)
    
    STOP.append(tokenizer(".").input_ids)  # stop decoding when seeing '.'
    SURE.append(tokenizer("sure").input_ids)
    UNSURE.append(tokenizer("unsure").input_ids)
    
    # Load test data
    data = []
    with open(f"../../dataset/pararel/{args.domain}_test_pararel.json", 'r') as f:
        data = json.load(f)
    
    # Load the trained classifier
    print(f"Loading classifier from {args.classifier_path}...")
    # First, we need to determine the input dimension by extracting features from a sample
    sample_feature = extract_features(model, tokenizer, data[0][0], args.layer_idx)
    input_dim = sample_feature.shape[0]
    
    # Initialize and load the classifier
    classifier = SAPLMAClassifier(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.load_state_dict(torch.load(args.classifier_path, map_location=device))
    classifier.to(device)
    classifier.eval()
    
    # Evaluate on test data
    with accelerator.split_between_processes(data) as data:
        results = []
        
        for sample in tqdm(data):
            # Perform inference
            output, full_input = inference(sample[0])
            
            # Determine if the answer is correct (ground truth)
            is_correct = sample[1] in output
            
            # Extract features for the classifier
            feature = extract_features(model, tokenizer, sample[0], args.layer_idx)
            
            # Use the classifier to predict correctness
            with torch.no_grad():
                feature_tensor = torch.FloatTensor(feature).unsqueeze(0).to(device)
                prediction_score = classifier(feature_tensor).item()
                prediction = 1 if prediction_score > 0.5 else 0
            
            # Store results: (actual correctness, classifier prediction, classifier score)
            result = (int(is_correct), prediction, prediction_score)
            results.append(result)
    
    results = gather_object(results)
    
    if accelerator.is_main_process:
        os.makedirs("results", exist_ok=True)
        with open(f"results/ours_{args.domain}_saplma_{model_name}.json", 'w') as f:
            json.dump(results, f)
        
        # Calculate and print metrics
        actual = [r[0] for r in results]
        predicted = [r[1] for r in results]
        
        correct_predictions = sum(1 for a, p in zip(actual, predicted) if a == p)
        total_predictions = len(actual)
        accuracy = correct_predictions / total_predictions
        
        print(f"Evaluation completed. Results saved to results/ours_{args.domain}_saplma_{model_name}.json")
        print(f"Accuracy of SAPLMA classifier: {accuracy:.4f}")
