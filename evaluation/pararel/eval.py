import json
import numpy as np
from argparse import ArgumentParser

def eval_acc(results):
    if results:
        preds = [res[0] for res in results]
    else:
        return 0
    return sum(preds), len(preds), sum(preds)/len(preds)
    # return sum(preds)/len(preds)

def eval_ap(results):
    sorted_data = sorted(results, key=lambda x: x[-1], reverse=True)

    num_correct = sum(1 for d in sorted_data if d[0])
    if num_correct == 0:
        return 0
    cumulative_correct = sorted_data[0][0]
    ap_sum = 0

    for k in range(1, len(sorted_data)):
        if sorted_data[k][0]:
            precision = cumulative_correct / k
            recall_delta = 1 / num_correct
            cumulative_correct += 1
            ap_sum += precision * recall_delta

    return ap_sum

def bootstrap_confidence_interval(data, num_bootstrap_samples=100000, confidence_level=0.95):
    """
    Calculate the bootstrap confidence interval for the mean of 1D accuracy data.
    Also returns the median of the bootstrap means.
    
    Args:
    - data (list or array of float): 1D list or array of data points.
    - num_bootstrap_samples (int): Number of bootstrap samples.
    - confidence_level (float): The desired confidence level (e.g., 0.95 for 95%).
    
    Returns:
    - str: Formatted string with 95% confidence interval and median as percentages with one decimal place.
    """
    # Convert data to a numpy array for easier manipulation
    data = np.array(data)

    # List to store the means of bootstrap samples
    bootstrap_means = []

    # Generate bootstrap samples and compute the mean for each sample
    for _ in range(num_bootstrap_samples):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # Compute the mean of the bootstrap sample
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)

    # Convert bootstrap_means to a numpy array for percentile calculation
    bootstrap_means = np.array(bootstrap_means)

    # Compute the lower and upper percentiles for the confidence interval
    lower_percentile = (1.0 - confidence_level) / 2.0
    upper_percentile = 1.0 - lower_percentile
    ci_lower = np.percentile(bootstrap_means, lower_percentile * 100)
    ci_upper = np.percentile(bootstrap_means, upper_percentile * 100)

    # Compute the median of the bootstrap means
    median = np.median(bootstrap_means)

    # Convert to percentages and format to one decimal place
    ci_lower_percent = ci_lower
    ci_upper_percent = ci_upper
    median_percent = median

    # Return the formatted string with confidence interval and median
    print(f"95% Bootstrap Confidence Interval: ({ci_lower_percent*100:.2f}%, {ci_upper_percent*100:.2f}%), Median: {median_percent*100:.2f}%")
    print(ci_upper_percent*100-median_percent*100)
    return f"95% Bootstrap Confidence Interval: ({ci_lower_percent*100:.2f}%, {ci_upper_percent*100:.2f}%), Median: {median_percent*100 :.2f}%"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--domain',type=str, default="ID")
    parser.add_argument("--num_try",type=int,default=10)
    parser.add_argument("--alpha",type=float,default=0.5)
    parser.add_argument("--tau",type=float,default=-0.8018185525433373)
    parser.add_argument("--model",type=str,default="openlm-research/open_llama_3b")
    parser.add_argument("--method",type=str,default="vanilla",choices=["vanilla","semantic","rtuning","classifier","kernel","gpt","clm","claude","scgpt","gpt_scgpt","claude_scgpt","saplma",'udeg'])
    args = parser.parse_args()
    model_name = args.model.split('/')[-1]

    if args.method == "clm":
        with open(f"results/ours_ID_clm_{model_name}.json",'r') as f:
            data = json.load(f)
    elif args.method == "claude_scgpt":
        with open(f"results/ours_ID_claude_scgpt.json",'r') as f:
            data = json.load(f)
    elif args.method == "gpt_scgpt":
        with open(f"results/ours_ID_gpt_4o_modified_scgpt.json",'r') as f:
            data = json.load(f)
    elif args.method == "claude":
        with open(f"results/ours_ID_claude.json",'r') as f:
            data = json.load(f)
        with open(f"results/ours_ID_15_kernel_open_llama_7b.json",'r') as f:
            certainty = json.load(f)
    elif args.method == "gpt":
        with open(f"results/ours_ID_gpt_4o.json",'r') as f:
            data = json.load(f)[:4890]
        with open(f"results/ours_ID_gpt_4o_modified.json",'r') as f:
            data = data+json.load(f)
        with open(f"results/ours_ID_15_semantic_open_llama_7b.json",'r') as f:
            certainty = json.load(f)
    elif args.method == 'rtuning' or args.method == 'classifier':
        with open(f"results/ours_{args.domain}_{args.method}_{model_name}.json",'r') as f:
            data = json.load(f)
    elif args.method == 'saplma':
        with open(f"results/ours_{args.domain}_saplma_{model_name}.json",'r') as f:
            data = json.load(f)
    else:
        with open(f"results/ours_{args.domain}_{args.num_try}_{args.method}_{model_name}.json",'r') as f:
            data = json.load(f)
    
    tau = args.tau

    uncertain_results = []
    certain_results = []
    if args.method == 'gpt' or args.method == 'claude':
        for i in range(len(data)):
            if certainty[i][-1] > tau:
                certain_results.append([data[i]])
            else:
                uncertain_results.append([data[i]])
        data = certain_results + uncertain_results
    else:
        for d in data:
            if args.method == 'scgpt' or args.method == 'gpt_scgpt' or args.method == 'claude_scgpt':
                d[-1] = -d[-1]
            if d[-1] > tau:
                certain_results.append(d)
            else:
                uncertain_results.append(d)

    # beta = 1

    # results = []
    # for d in data:
    #     predict_conf, sure_conf = d[-2], d[-1]
    #     conf = beta*sure_conf + (1-beta)*predict_conf
    #     updated_result = list(d) + [conf]
    #     results.append(updated_result)
    
    print('All:')
    certain_num, total_num, total_acc = eval_acc(data)
    uncertain_num = total_num-certain_num
    print(certain_num, total_num, total_acc)
    print(eval_ap(data))

    print('Certain:')
    certain_num2, total_num2, total_acc2 = eval_acc(certain_results)
    print(certain_num2, total_num2, total_acc2)
    print(eval_ap(certain_results))
    
    # Calculate bootstrap confidence interval for certain results
    if certain_results:
        certain_preds = [res[0] for res in certain_results]
        bootstrap_confidence_interval(certain_preds)

    print('Uncertain:')
    certain_num3, total_num3, total_acc3 = eval_acc(uncertain_results)
    print(certain_num3, total_num3, total_acc3)
    print(eval_ap(uncertain_results))

    print('fnr', (total_num2-certain_num2)/uncertain_num)
    print('fpr', certain_num3/certain_num)