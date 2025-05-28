from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
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

STOP = []
SURE = []
UNSURE = []

cache_dir = '/work/vita/nie/cache/huggingface/hub'
end_chars = ['.', '\n']
llh_shift = torch.tensor(5.0)
# device=torch.device("cuda")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def format_question(input_data):
    
    context_ls = []
    for single_context in input_data['context']:
        context_ls.append(single_context[0] + ":" + " ".join(single_context[1]) + "\n")
    context_str = " ".join(context_ls)
    full_input = context_str + "\nQuestion: " + input_data['question'] + "\nAnswer:" 
    return full_input

def inference(input_text, model, tokenizer):
    full_input = format_question(input_text)
    inputs = tokenizer(full_input,return_tensors="pt").to(device)
    ids = inputs['input_ids']
    length = len(ids[0])     
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
        
    return output_text, full_input, np.power(product.item(),(1/count)).item()


def calculate_certainty(input_data):
    # Parameter settings
    weight_entailment = args.weight_entail  # Weight for the entailment class
    weight_neutral = args.weight_neutral  # Weight for the neutral class
    t = args.time  # Heat kernel diffusion time parameter

    # Initialize variables
    answers = []
    full_input = format_question(input_data)
    question = input_data['question']
    inputs = tokenizer(full_input, return_tensors="pt").to(device)
    ids = inputs['input_ids']
    length = len(ids[0])

    # Store generated sequences
    generations = torch.ones((args.num_try, length + 15), dtype=torch.long, device=device)
    
    # Generation phase: using your provided code
    for i in range(args.num_try):
        generation = model.generate(
            ids,
            num_return_sequences=1,
            num_beams=1,
            temperature=0.7,
            do_sample=True,
            max_new_tokens=15,
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


def checksure(input_text):
    full_input = f"{input_text}. Are you sure you accurately answered the question based on your internal knowledge? I am"
    inputs = tokenizer(full_input,return_tensors="pt").to(0)
    ids = inputs['input_ids']
    outputs = model.generate(
                ids,
                max_new_tokens = 1,
                output_scores = True,
                return_dict_in_generate=True
            )
    logits = outputs['scores']
     #greedy decoding and calculate the confidence of sure and unsure
    pt = torch.softmax(torch.Tensor(logits[0][0]),dim=0)
    sure_prob = pt[SURE[0]]
    unsure_prob = pt[UNSURE[0]]
    sure_prob = sure_prob/(sure_prob+unsure_prob)   #normalization
       
    return sure_prob.item()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='openlm-research/open_llama_3b')
    parser.add_argument('--assist_model', type=str, default='Qwen/Qwen2.5-3B')
    parser.add_argument('--result',type=str, default="Hotpot")
    parser.add_argument("--num_try",type=int,default=5)
    parser.add_argument("--alpha",type=float,default=0.5)
    parser.add_argument('--beta',type=float,default=1)
    parser.add_argument("--tau",type=float,default=0.5)
    parser.add_argument('--scale', type=str, default='3b')
    parser.add_argument('--seed', type=int, default=999, help='random seed')
    parser.add_argument('--weight_entail', type=float, default=0.7, help='Weight for the entailment class')
    parser.add_argument('--weight_neutral', type=float, default=0.3, help='Weight for the neutral class')
    parser.add_argument('--time', type=float, default=0.5, help='Heat kernel diffusion time parameter')

    args = parser.parse_args()
    model_name = args.model.split('/')[-1]
    set_seed(args.seed)
    accelerator = Accelerator()
    device = accelerator.device
    
    
    # tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False,cache_dir=cache_dir)
    # model = AutoModelForCausalLM.from_pretrained(args.model,device_map='auto',torch_dtype=torch.float16,cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False,cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model,device_map=device,torch_dtype=torch.float16,cache_dir=cache_dir)
    # assist_tokenizer = AutoTokenizer.from_pretrained(args.assist_model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False,cache_dir=cache_dir)
    # assist_model = AutoModelForCausalLM.from_pretrained(args.assist_model,device_map='auto',torch_dtype=torch.float16,cache_dir=cache_dir)
    
    deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli",cache_dir=cache_dir)
    deberta_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v2-xlarge-mnli",cache_dir=cache_dir).cuda()

    STOP.append(tokenizer(".").input_ids)  #stop decoding when seeing '.'
    SURE.append(tokenizer("sure").input_ids)
    UNSURE.append(tokenizer("unsure").input_ids)
    THRESHOLD = args.tau

    with open(f"../../dataset/HotpotQA/hotpot_test.json",'r') as f:
        data = json.load(f)


    with accelerator.split_between_processes(data) as data:
        results=[]

        for sample in tqdm(data):
            output, full_input, predict_conf = inference(sample, model, tokenizer)
            # assist_output, assist_full_input, assist_predict_conf = inference(sample,assist_model, assist_tokenizer)
            
            # print(f"Output: {output}, Full Input: {full_input}, Prediction Confidence: {predict_conf}")
            
            # print(f"This is sample {sample}\n______________________")
            certainty = calculate_certainty(sample)
            #sure_prob = checksure(f"{full_input} {output}")
            result = (sample['answer'].lower() in output.lower(), predict_conf, certainty.item())
            print(sample['answer'].lower() in output.lower(), certainty.item())
            results.append(result)

    results=gather_object(results)
    
    if accelerator.is_main_process:
        os.makedirs("results",exist_ok=True)
        with open(f"results/ours_{args.num_try}_kernel_{model_name}_{args.weight_entail}_{args.weight_neutral}_{args.time}.json",'w') as f:
            json.dump(results,f)

                 
