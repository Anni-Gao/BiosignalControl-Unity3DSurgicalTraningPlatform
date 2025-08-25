# Biosignal Control for Unity3D Surgical Training Platform

## Introduction
This project provides a bio-signal–driven control system integrated into a Unity3D-based surgical training platform. The system enables real-time analysis of user cognitive and physiological states, providing insights into performance, stress, and learning outcomes in surgical skill training.

## Features

### Bio-signal Data Range Calibration & Real-time Collection
EEG (attention, stress) - Cortex API connect

HRV (RMSSD) - bluetooth connect

Action events (move / stop detection)

Unity3D condition (mode) control - UDP transfer
### Python Data Analysis
Performance Metrics : MAE, completion time, pause counts, etc.

Biosignal Integration: Aligns EEG and HRV data with trajectory timelines to study attention, stress, and physiological responses.

Visualization: Generates table and diagrams to compare and evaluate results.

## Equipments
HRV : [POLAR H10 chest band](https://www.polar.com/uk-en/sensors/h10-heart-rate-sensor)

EEG : [EMOTIV MN8 earphone](https://www.emotiv.com/products/mn8)

## Develop Environment
Pycharm (Python 3.9.6 version)

`pip install -r requirements.txt`

## Project Structure
```
├── analyse/                 # Analysis scripts
│   ├── output/              # Results (users / summary / HRV-EEG analysis)
│   │   ├── users/           # Per-user analysis results
│   │   ├── summary/         # Aggregated results
│   │   └── hrv_eeg_analysis/# HRV & EEG overlay analysis
│   ├── dataset/
│   │   ├── range/           # HRV & EEG range
│   │   └── user_data/       # Bio-signals & trajectories logs
├── eeg/
│   ├── eeg_user_utils.py
│   ├── eeg_utils.py
│   ├── log_eeg_data.py
│   └── main_eeg.py          # EEG calibration & collection
├── hrv/
│   ├── hrv_user_utils.py
│   ├── hrv_utils.py
│   ├── log_hrv_data.py
│   └── main_hrv.py          # HRV calibration & collection
├── unity/
│   ├── BiosignalScalpelController.cs # Unity3D C# script
│   └── unity_server.py      # Unity3D communication server
├── requirements.txt         # Python dependencies
└── README.md
```


## Usage
### Data Preparing
1. Extract the compressed file into the folder **dataset**  
2. Move them into the folder **analyse**, at the same level as the `.py` files
(Only Condition A & Condition B dataset are used in this project)

### Run System
1. Run `main.eeg.py` & `main.hrv.py` files to start range calibration and collect real-time data.  
2. Run `unity_server.py` file and insert a number to change the control mode:  
   - **Mode 1**: Without any bio-feedback (baseline compare)  
   - **Mode 2**: HRV control (once out of RMSSD range, the scalpel disappears)  
   - **Mode 3**: EEG control (once attention falls below `attention_low`, the scalpel disappears)  

### Data Analysis
Run all `.py` files in the folder **analyse** to perform data analysis.  

### Output
- **Per-user results** → `analyse/output/users/`  
- **Aggregated results** → `analyse/output/summary/`  
- **Alternative results** → `analyse/output/hrv_eeg_analysis/` 
