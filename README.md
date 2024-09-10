This code repository is a part of my Master internship research, which developes a machine learning model (Transformer) to learn and predict path flow distribution at optimal (User equilibrium) state. 
This research is submited for a presentation at TRB Annual Conference 2025.

The best model so far:
- Input features:
    + Graph characteristics: Link length and free flow travel time => Non-normal distribution => Use log1p normalize
    + OD demand: Unify distribution => Use MinMax Normalize
    + 3 fesible paths: select top 3 shortest paths for each OD pair, label encode to get path ID => Use MinMax Normalize.
- Output:
    Include 3 path flows corresponding to 3 paths for each OD pair. Total demand of this OD pair will be distributed into these 3 paths => Use MinMax Normalize for each OD pair (normalize by row)
- Model architecture: add 1 more layerNorm layer after each skip connection, in both Encoder and Decoder.
- Result: Path flow MAPE: 4.19%, Link flow MAPE: 2.91%

The model can predict path flow at optimal state when randomly remove 2 or 3 links without retrain the model (make sure there is at least 1 path for each OD pair in the matrix).
