# ELP-Mob

Code for **"Building Efficient LLM Pipeline for Human Mobility Prediction"**

This repository provides an end-to-end pipeline for **human mobility prediction** using **LLMs**.  

---

## 📊 Dataset and Model

- **Dataset**  
  The benchmark dataset comes from [GISCUP 2025](https://sigspatial2025.sigspatial.org/giscup/dataset.html).  
  Please download it and place the files under: <tt>../sigspatial_datasets/</tt>

- **Model**  
  The **Llama-3.2-3B-Instruct** model is used in our experiments.  
  You can download it from Hugging Face: [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)  
  Place the model files under: <tt>../Llama-3.2-3B-Instruct/</tt>

---

## ⚙️ Installation

Set up the environment with **conda**:

```bash
conda create --name ELP-Mob python=3.10
conda activate ELP-Mob
```

Clone and install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory):

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

Clone and install [GEO-BLEU](https://github.com/yahoojapan/geobleu):
```bash
git clone https://github.com/yahoojapan/geobleu.git
cd geobleu
pip3 install .
```

---

## 🚀 Usage

### 1. Preprocess the dataset
```bash
python preprocess.py
```

---

### 2. Generate prompts for LLMs
```bash
python gen_prompts.py --dataset {city_name}
```

Available `{city_name}` options: `A`  `B` `C` `D`  

---

### 3. Fine-tune the model & validate
Run baselines, fine-tune the model, and evaluate on the validation set:

```bash
python baselines.py --dataset {city_name} --mode val
llamafactory-cli train ./configs/sft_{city_name}.yaml
llamafactory-cli train ./configs/predict_{city_name}.yaml
python predictions_to_csv.py --dataset {city_name} --mode val
python evaluate.py --dataset {city_name} --method llm --mode val
```

---

### 4. Perform prediction on the test set
```bash
python baselines.py --dataset {city_name} --mode test
llamafactory-cli train ./configs/test_{city_name}.yaml
python predictions_to_csv.py --dataset {city_name} --mode test
```
