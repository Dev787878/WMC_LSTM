# WMC-LSTM Project

## Overview
This project implements an improved version of Long Short-Term Memory (LSTM) using **Working Memory Connections (WMC)**.

The goal is to enhance sequence learning by allowing the internal cell state to directly influence the gating mechanism, leading to better performance compared to vanilla LSTM.

---

## ⚙️ Tech Stack
- Python  
- PyTorch  
- Matplotlib  

---

## Models Implemented
- **Baseline:** Vanilla LSTM  
- **Modified:** WMC-LSTM (with cell-to-gate connections)  

---

## Experiments

### 1. Regression
- Input: Sequence of numbers  
- Output: Sum of sequence  

### 2. Classification
- Input: Sequence  
- Output: 1 if sum > 0, else 0  

---

## Setup & Installation

### 1. Create virtual environment
python -m venv venv

### 2. Activate environment

Windows:
venv\Scripts\activate

Linux/Mac:
source venv/bin/activate

### 3. Install dependencies
pip install -r requirements.txt

---

## How to Run

Run Regression:
python train_reg.py

Run Classification:
python train_cls.py

Run Comparison:
python train_compare.py

---

## 📈 Outputs
- loss.png → Loss comparison graph  
- accuracy.png → Accuracy comparison graph  

---
