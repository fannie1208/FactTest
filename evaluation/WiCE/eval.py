import json
from argparse import ArgumentParser

def eval_acc(results):
    if results:
        preds = [res[0] for res in results]
    else:
        return 0, 0, 0
    return sum(preds), len(preds), sum(preds)/len(preds)
    # return sum(preds)/len(preds)

def eval_ap(results, base=False):
    if base:
        sorted_data = sorted(results, key=lambda x: x[-3], reverse=True)
    else:
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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_try",type=int,default=10)
    parser.add_argument("--tau",type=float,default=-0.5)
    parser.add_argument("--model",type=str,default="openlm-research/open_llama_3b")
    parser.add_argument("--method",type=str,default="vanilla",choices=["vanilla","semantic"])
    args = parser.parse_args()

    model_name = args.model.split('/')[-1]
    with open(f"results/ours_{args.num_try}_{args.method}_{model_name}.json",'r') as f:
        data = json.load(f)
    
    tau = args.tau

    uncertain_results = []
    certain_results = []


    for d in data:
        if d[-1] > tau:
            certain_results.append(d)
        else:
            uncertain_results.append(d)

    
    print('All:')
    certain_num, total_num, total_acc = eval_acc(data)
    uncertain_num = total_num-certain_num
    print(certain_num, total_num, total_acc)

    print('Certain:')
    print(len(certain_results))
    certain_num2, total_num2, total_acc2 = eval_acc(certain_results)
    print(certain_num2, total_num2, total_acc2)
    print(eval_ap(certain_results))

    print('Uncertain:')
    certain_num3, total_num3, total_acc3 = eval_acc(uncertain_results)
    print(certain_num3, total_num3, total_acc3)

    print('fnr', (total_num2-certain_num2)/uncertain_num)
    print('fpr', certain_num3/certain_num)