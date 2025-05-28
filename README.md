# 🧪 FactTest: Factuality Testing in Large Language Models with Finite-Sample and Distribution-Free Guarantees

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2411.02603-b31b1b.svg)](https://arxiv.org/abs/2411.02603)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


</div>

---

## 👥 Authors

<table>
<tr>
<td align="center">
<a href="https://scholar.google.com/citations?user=o2lsU8YAAAAJ&hl=en">
<strong>Fan Nie</strong>
</a>
</td>
<td align="center">
<strong>Xiaotian Hou</strong>
</td>
<td align="center">
<strong>Shuhang Lin</strong>
</td>
<td align="center">
<a href="https://www.james-zou.com/">
<strong>James Zou</strong>
</a>
</td>
<td align="center">
<a href="https://www.huaxiuyao.io/">
<strong>Huaxiu Yao</strong>
</a>
</td>
<td align="center">
<a href="https://linjunz.github.io/index.html">
<strong>Linjun Zhang</strong>
</a>
</td>
</tr>
</table>

---

## 📰 News

- **🎉 May 27, 2025**: Source code released!

---

## 🚀 Quick Start

This repository provides tools for testing factuality in Large Language Models with statistical guarantees. Follow the steps below to get started with ParaRel as an example.

### 📋 Prerequisites

```bash
# Clone the repository
git clone https://github.com/your-username/FactTest.git
cd FactTest

# Install dependencies (add your requirements.txt)
pip install -r requirements.txt
```

---

## 🔧 Usage

### 🎯 Step 1: Calibration

Navigate to the calibration directory:
```bash
cd calibration/pararel
```

#### 📊 Calibration Dataset Construction
```bash
python collect_dataset.py --model openlm-research/open_llama_3b
```

#### 🎚️ Calibration and Threshold Selection
```bash
python calculate_vanilla_threshold.py \
    --model openlm-research/open_llama_3b \
    --alpha 0.05 \
    --num_try 15
```

### 📈 Step 2: Evaluation

```bash
cd evaluation/pararel
python evaluate_vanilla.py \
    --model openlm-research/open_llama_3b \
    --num_try 15
```

#### 📊 Calculate Evaluation Metrics
After evaluation, compute the metrics using:
```bash
python eval.py \
    --model openlm-research/open_llama_3b \
    --num_try 15 \
    --method vanilla \
    --tau <your_threshold>
```

> **💡 Note**: Replace `<your_threshold>` with the threshold value obtained from the calibration step.

---


---

## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{nie2024facttest,
      title={FactTest: Factuality Testing in Large Language Models with Finite-Sample and Distribution-Free Guarantees}, 
      author={Fan Nie and Xiaotian Hou and Shuhang Lin and James Zou and Huaxiu Yao and Linjun Zhang},
      year={2024},
      eprint={2411.02603},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.02603}, 
}
```

