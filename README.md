<div align="center">
  <img src="Figs/envscaler_logo.png" width="150px">
</div>
<h1 align="center"> EnvScaler: Scaling Tool-Interactive Environments for LLM Agent via Programmatic Synthesis</a></h1>


<div align="center">
  <a href="https://arxiv.org/abs/TODO">
    <img src="https://img.shields.io/badge/Paper-arXiv-b5212f.svg?logo=arxiv" alt="Arxiv">
  </a>
  <a href="https://huggingface.co/collections/XXHStudyHard/envscaler">
    <img src="https://img.shields.io/badge/Model-Hugging%20Face-blue?logo=huggingface" alt="Hugging Face Models">
  </a>
  <a href="https://huggingface.co/collections/XXHStudyHard/envscaler">
    <img src="https://img.shields.io/badge/Dataset-Hugging%20Face-blue?logo=huggingface" alt="Hugging Face Datasets">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/LICENSE-MIT-green.svg" alt="License">
  </a>
  <a href="https://www.python.org/downloads/release/python-312/">
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  </a>
</div>

## 🎬 Demo

### Env-Agent-User Interaction
<div align="center">
    <video src="https://github.com/user-attachments/assets/613b46fd-63db-4050-91d2-f7aca2a766e3" />
</div>

### Env-Agent Interaction
<div align="center">
    <video src="https://github.com/user-attachments/assets/b8186257-a22d-4ec1-9ccf-82f6bd23a4b5" />
</div>

### Building Environment From Scratch
<div align="center">
    <video src="https://github.com/user-attachments/assets/fd947e46-014a-41cd-87bb-6744c3dd5b32" />
</div>

To locally run the demo that interacting with Envs:
```bash
cd interact_with_env
python app.py
```
To locally run the demo that builing Envs from scratch:
```bash
cd skel_builder
python env_build_demo.py
```

## 👀 Overview

**EnvScaler** is an automated framework that mass-produces executable, stateful, tool-based environments for training large-language-model agents. 
<p align="center">
    <img src="Figs/envscaler_overview.png" width="60%"> <br>
  Overview of <b>EnvScaler</b>.
</p>

**SkelBuilder** is the first stage of EnvScaler. Starting from existing open-source text tasks, it mines latent domain descriptions, plans the corresponding state schema and business rules, and generates a fully functional Python class whose methods expose tool interfaces. A dual-agent loop (one proposes random tool calls, the other inspects code, return values, and state changes) keeps only classes that pass the majority of test rounds, ensuring quality and consistency.
<p align="center">
    <img src="Figs/skelbuilder_framework.png" width="98%"> <br>
  Framework of <b>SkelBuilder</b>.
</p>

**ScenGenerator** is the second stage. Given an approved skeleton, it first prompts the LLM to generate a initial state/database, then creates a challenging user task that can be solved from that state. Finally, it decomposes the task into independent checkpoints and converts each into a Python boolean function over the final state, providing an automatic reward signal and enabling trajectory evaluation regardless of the agent’s solution path.
<p align="center">
    <img src="Figs/scengenerator_framework.png" width="98%"> <br>
  Framework of <b>ScenGenerator</b>.
</p>

## 📊 Results
TODO

## 🚀 Quick Start
TODO




## 📚 Citation

If you find our work helpful, please consider citing it. We greatly appreciate your support.

```bibtex
TODO
```