This code repository is a part of my Master internship research, which developes a machine learning model (Transformer) to learn and predict path flow distribution at optimal (User equilibrium) state. 
This research is accepted for a presentation at TRB Annual Conference 2025.

If you use this code for your research, please cite it:\
_V.A. Le, M. Ameli, and A. Skabardonis. A machine learning approach for network equilibrium estimation. The 104th Transportation Research Board Annual Meeting (TRB 2025), Jan 2025, Washington D. C., United States._

@inproceedings{le:hal-04750911, \
TITLE = {{A Machine Learning Approach for Network Equilibrium Estimation}}, \
AUTHOR = {Le, Van Anh and Ameli, Mostafa and Skabardonis, Alexander}, \
URL = {https://hal.science/hal-04750911}, \
BOOKTITLE = {{The 104th Transportation Research Board Annual Meeting (TRB 2025)}}, \
ADDRESS = {Washington D. C., United States}, \
YEAR = {2025}, \
MONTH = Jan, \
HAL_ID = {hal-04750911}, \
HAL_VERSION = {v1}, }

# A MACHINE LEARNING APPROACH FOR NETWORK EQUILIBRIUM ESTIMATION

## FOLDERS DESCRIPTION & HOW TO USE:
1. Generate_data
- Contain network file and demand file of Sioux Falls, EMA, and Synthetic network (folder Random)
- File Dataset_Generation: 
    - To generate synthetic network 
    - To generate OD demand matrix 
    - To solve UE solution with Gurobi
- File utils: contains functions to read network, solve UE solution
- File check_UE: to verify the solution solved by Gurobi, draw charts

2. Solution \
Contains UE solution solved by Gurobi for:
- Synthetic network (Random):
    - Random_30: solution of full links, missing 30% demand
    - Random_40: solution of full links, missing 40% demand
- Sioux Falls network:
    - MultiClass: solution of Multi-class network
    - Output1: solution of Single-class, full links, missing 30% demand
    - Output2: solution of Single-class, missing 5% link, missing 30% demand
    - Output3: solution of Single-class, missing 10% links, missing 30% demand
- EMA: solution of full links, missing 50% demand

3. Model \
Contains the code of pre-processing data, transformer, and evaluating model.
- multi: Code for multi-class network
- single: Code for single-class network
- multi_main.py: to run training and predicting for multi class network
- single_main.py: for single network
- plotting.py: plotting charts 

## INSTALLATION
Step-by-step guide on how to install and set up your project.
1. Installation:
- Make sure you have at least 43GB of empty disk to store this project.
- Install Cuda 12.3
- Install CuNN 8.9.7
- Install Anaconda 3
- Create virtual environment and install tensorflow:
Open Anaconda Powershell Prompt, run:
    ```
    conda create --name gpu_env python=3.9
    conda activate gpu_env
    conda install tensorflow
    ```

2. Clone the repository
Open cmd in Windows / Terminal in Mac
```
git clone https://github.com/VivianeLe/Path_Flow_Prediction.git
```

3. Install dependencies:
Open Anaconda Powershell Prompt, run:
```
pip install -r requirements.txt
```

4. Running 
- Check file Dataset_Generation.ipynb if you want to re-generate new demand. Then resolve the UE solution with your new demand 
- Train your model:
    - For single-class network: 
        - Edit the parameters in file Model/single/single_params.py
        - In Anaconda Powershell Prompt, cd to folder Model, run: ```python single_main.py```
    - For multi-class network: 
        - Edit the parameters in file Model/multi/multi_params.py
        - In Anaconda Powershell Prompt, cd to folder Model, run: ```python multi_main.py```

