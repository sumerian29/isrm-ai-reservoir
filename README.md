# 🛢️ ISRM Advanced Stable
### AI-Based Reservoir Management System for Iraqi Oil Fields

---

## 📌 Overview

**ISRM (Iraqi Smart Reservoir Manager)** is a practical AI-based decision support system designed to enhance reservoir management in Iraqi oil fields.

The system integrates data analytics, machine learning, and engineering logic to support production optimization, anomaly detection, and forecasting.

---

## 🎯 Objectives

- Improve oil production without additional CAPEX
- Reduce water cut and operational inefficiencies
- Detect abnormal well behavior early
- Support data-driven decision-making in reservoir management

---

## 🚀 Key Features

### 📊 1. Interactive Dashboard
- Real-time visualization of:
  - Total wells
  - Field production
  - Average BHP
  - Water cut
  - Critical wells

---

### 🔍 2. Well Performance Analysis
- Detailed well-level analysis
- Trend monitoring (production, pressure, water cut)

---

### 🧠 3. AI-Based Forecasting

Supports multiple machine learning models:

- **Random Forest** → robust and stable for small datasets  
- **XGBoost** → high accuracy and advanced learning  
- **LSTM** → time-series deep learning model  

---

### ⚙️ 4. Production Optimization
- Suggests optimal production rates
- Supports field-level production strategy

---

### 🚨 5. Anomaly Detection
- Identifies abnormal well behavior
- Early detection of:
  - Sudden production drop
  - Pressure anomalies
  - Water breakthrough

---

### 🧪 6. Well Integrity Index (WII)
- Classifies wells into:
  - Healthy
  - Watch
  - Critical

Based on:
- Production rate
- Pressure
- Water cut
- Stability

---

### 📤 7. Export Capability
- Export results to:
  - Excel
  - PDF reports

---

## ⚙️ System Architecture

The system consists of:

1. Data Input (Excel / CSV)
2. Data Processing & Cleaning
3. Machine Learning Models
4. Visualization Layer (Streamlit)
5. Decision Support Outputs

---

## 🧠 How It Works

1. Upload well data (or use synthetic dataset)
2. Select forecasting model (Auto / RF / XGB / LSTM)
3. Analyze field and well performance
4. Generate predictions and optimization insights

---

## 📊 Required Data Format

| Column Name     | Description               |
|----------------|--------------------------|
| Date           | Production date          |
| Well_Name      | Well identifier          |
| Oil_Rate       | Oil production rate      |
| Water_Cut      | Water cut (%)            |
| BHP            | Bottom-hole pressure     |
| THP            | Tubing head pressure     |
| Permeability   | Reservoir permeability   |
| Net_Pay        | Net pay thickness        |

Optional:
- Intervention
- Well Status

---

## ▶️ Installation

```bash
pip install -r requirements.txt# ISRM Advanced Stable

Iraqi Smart Reservoir Manager – Practical AI Reservoir Decision Support System.

## Overview

ISRM is a Streamlit-based prototype for AI-assisted reservoir management in Iraqi oilfields.  
It analyzes well performance, calculates a Well Integrity Index, detects anomalies, forecasts production, and suggests optimized production rates.

## Features

- Interactive dashboard
- Well performance analysis
- Well Integrity Index (WII)
- Production forecasting
- Rate optimization
- Anomaly detection
- Excel and PDF export

## Installation

```bash
pip install -r requirements.txt
