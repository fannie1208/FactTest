from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import DebertaTokenizer, DebertaForSequenceClassification
from accelerate import Accelerator
from accelerate.utils import gather_object
import torch
import torch.nn.functional as F
import json
from tqdm.auto import tqdm
import random
from argparse import ArgumentParser
from scipy.stats import entropy
import math
import os
import numpy as np
import evaluate
from eval import eval_acc, eval_ap



# Set a seed value
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

end_chars = ['.', '\n']
# device='cuda'
llh_shift = torch.tensor(5.0)

# os.environ['NCCL_TIMEOUT'] = '3600'
# os.environ['NCCL_DEBUG'] = 'INFO'

def calculate_certainty(input_data):

    weight_entailment = args.weight_entail  # Weight for the entailment class
    weight_neutral = args.weight_neutral  # Weight for the neutral class
    t = args.time  # Heat kernel diffusion time parameter

    answers = []
    occurance = {}
    full_input = "Question:" + input_data + " Answer:"
    question = input_data
    inputs = tokenizer(full_input,return_tensors="pt").to(device)
    ids = inputs['input_ids']
    length = len(ids[0])
    generations = torch.ones((args.num_try, length + 15),
                                     dtype=torch.long,
                                     device=device)

    for i in range(args.num_try):
        generation = model.generate(
            ids,
            num_return_sequences=1,
            num_beams=1,
            temperature=0.7,
            do_sample = True,
            max_new_tokens = 15,
        )
        generations[i, :generation.shape[1]] = generation
        output_text = tokenizer.decode(generation[0][length:])
        idx = min([output_text.find(char) for char in end_chars if output_text.find(char) != -1] + [len(output_text)])
        output_text = output_text[:idx]
        answers.append(output_text)
    
    # Get unique generated texts
    unique_generated_texts = list(set(answers))
    
    # If only one unique answer is generated, directly return entropy as 0
    if len(unique_generated_texts) == 1:
        print("von neumann entropy: 0")
        return torch.tensor(0.0)
    
    # Initialize similarity matrix
    num_unique_texts = len(unique_generated_texts)
    similarity_matrix = torch.zeros((num_unique_texts, num_unique_texts), device=device)

    # Batch computation of similarities
    for i, qa_1 in enumerate(unique_generated_texts):
        for j in range(i + 1, num_unique_texts):
            qa_2 = unique_generated_texts[j]

            # Forward concatenation
            ori_input = qa_1 + ' [SEP] ' + qa_2
            encoded_input = deberta_tokenizer(ori_input, return_tensors="pt", padding=True).to(device)
            prediction = deberta_model(**encoded_input)['logits']
            predicted_probs = F.softmax(prediction, dim=1)
            forward_similarity = weight_entailment * predicted_probs[0, 2] + weight_neutral * predicted_probs[0, 1]

            # Reverse concatenation
            reverse_input = qa_2 + ' [SEP] ' + qa_1
            encoded_reverse_input = deberta_tokenizer(reverse_input, return_tensors="pt", padding=True).to(device)
            reverse_prediction = deberta_model(**encoded_reverse_input)['logits']
            reverse_predicted_probs = F.softmax(reverse_prediction, dim=1)
            reverse_similarity = weight_entailment * reverse_predicted_probs[0, 2] + weight_neutral * reverse_predicted_probs[0, 1]

            # Compute the average similarity
            similarity = (forward_similarity + reverse_similarity) / 2
            distance = 1 - similarity.item()

            # Update the similarity matrix
            similarity_matrix[i, j] = similarity_matrix[j, i] = distance    

    # Semantic kernel computation (Heat Kernel)
    kernel_matrix = torch.exp(-t * similarity_matrix)

    # Trace normalization and scaling
    diag_values = torch.sqrt(torch.diag(kernel_matrix).unsqueeze(0))  # Compute the diagonal elements
    kernel_matrix = kernel_matrix / (diag_values.T @ diag_values) / num_unique_texts

    # Compute von Neumann entropy
    entropy = von_neumann_entropy(kernel_matrix)
    print(f"von neumann entropy: {entropy.item()}")

    return -entropy

def von_neumann_entropy(kernel_matrix):
    # Compute eigenvalues of the kernel matrix
    eigenvalues = torch.linalg.eigvalsh(kernel_matrix)
    # Filter out eigenvalues close to 0 to avoid log(0)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    # Compute von Neumann entropy
    entropy = -torch.sum(eigenvalues * torch.log(eigenvalues))
    return entropy

def inference(input_text, model, tokenizer):
    full_input = f"Question: {input_text} Answer:"
    inputs = tokenizer(full_input, return_tensors="pt").to(device)
    ids = inputs['input_ids']
    outputs = model.generate(
        ids,
        max_new_tokens=15,
        output_scores=True,
        return_dict_in_generate=True
    )
    logits = outputs['scores']
    output_sequence = []
    product = 1
    count = 0
    for i in logits:  # greedy decoding and calculate the confidence
        pt = torch.softmax(torch.Tensor(i[0]), dim=0)
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

    return output_text, full_input, np.power(product.item(), (1 / count)).item()

STOP = []
SURE = []
UNSURE = []

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default="openlm-research/open_llama_3b")
    parser.add_argument('--result', type=str, default="pararel")
    parser.add_argument('--domain', type=str, default="ID")
    parser.add_argument("--num_try", type=int, default=10)
    parser.add_argument("--alpha",type=float,default=0.5)
    parser.add_argument('--scale', type=str, default='3b')
    parser.add_argument('--seed', type=int, default=999, help='random seed')
    parser.add_argument('--weight_entail', type=float, default=0.7, help='Weight for the entailment class')
    parser.add_argument('--weight_neutral', type=float, default=0.3, help='Weight for the neutral class')
    parser.add_argument('--time', type=float, default=0.5, help='Heat kernel diffusion time parameter')
    args = parser.parse_args()
    model_name = args.model.split('/')[-1]
    set_seed(args.seed)
    # Initialize accelerator
    accelerator = Accelerator()

    device = accelerator.device
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, unk_token="<unk>", bos_token="<s>", eos_token="</s>", add_bos_token=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map=device, torch_dtype=torch.float16)

    deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
    deberta_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v2-xlarge-mnli").cuda()
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    period_token_id = tokenizer('. ')['input_ids'][1]
    eos_tokens = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
    question_framing_ids = [[tokenizer(eos_token)['input_ids'][1]] for eos_token in eos_tokens]

    STOP.append(tokenizer(".").input_ids)
    SURE.append(tokenizer("sure").input_ids)
    UNSURE.append(tokenizer("unsure").input_ids)

    data = []
    with open(f"../../dataset/pararel/{args.domain}_test_pararel.json", 'r') as f:
        data = json.load(f)

    with accelerator.split_between_processes(data) as data:
        results=[]

        for sample in tqdm(data):
            output, full_input, predict_conf = inference(sample[0], model, tokenizer)
            # assist_output, assist_full_input, assist_predict_conf = inference(sample[0], assist_model, assist_tokenizer)
            certainty = calculate_certainty(sample[0])
            result = (sample[1] in output, predict_conf, certainty.item())
            print(result)
            results.append(result)
        # results = [results]

    results_gathered=gather_object(results)

    if accelerator.is_main_process:

        # Save results to files
        os.makedirs("results", exist_ok=True)
        with open(f"results/ours_{args.domain}_{args.num_try}_kernel_{model_name}_{args.weight_entail}_{args.weight_neutral}_{args.time}.json",'w') as f:
            json.dump(results_gathered, f)


