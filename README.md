# PIML
The official PyTorch implementation of "Physics-infused Machine Learning for Crowd Simulation" (KDD'22)

To run the training, fine-tuning, and testing process and save the model, just run 
```
cd src
python main.py
```
The result will be saved in the `saved_model` folder. If you want to change the configuration, see the `get_args()` function in `src/main.py` for the available options. Here are some important options:
- `--finetune_flag` to conduct the finetuning process after training.
- `--data_config` to set the train, valid, and test dataset for training.
- `--ft_data_config` to set the train, valid, and test dataset for finetuning.

### Other Scripts
- `src/symbolic_regression.py` to conduct symbolic regression for the discovered MLAPM model.
- `src/main_mlapm.py` to run the simulation with the discovered MLAPM model.
- `src/run_experiments.py` to conduct grid searching for hyperparameters.

### Requirements
```
python            3.12.2
matplotlib        3.8.4
numpy             1.26.4
PyYAML            6.0.1
setproctitle      1.3.3
torch             2.2.2
tqdm              4.66.2
```

### About the Data

- **GC Dataset**: `./data/GC_Dataset/*.npy`, e.g., `GC_Dataset_ped1-12685_time1000-1060_interp9_xrange5-25_yrange15-35.npy`
  - `ped1-12685`: Select all 12685 pedestrians.
  - `time1000-1060`: Use the time range from 1000s to 1060s, duration=1min.
  - `interp9`: Interplate 9 extra steps between time steps, such that $\Delta t = 0.08s$.
  - `xrange5-25_yrange15-35`: Use the region where $5m \leq x \leq 25m, 15, \leq y \leq 35m$.
- **UCY Dataset**: `./data/UCY_Dataset/*.npy`, e.g., `UCY_Dataset_time0-54_timeunit0.08.npy`
  - `time0-54`: Use the time range from the initial to 54s
  - `timeunit0.08`: Inteplate along the time axis such that $\Delta t = 0.08s$.
- **Synthetic Dataset**: `./data/synthetic_data/*.npy`, e.g., `GC_Dataset_ped1-12685_time1560-1620_interp9_xrange5-25_yrange15-35_simulation.npy`
  -   `GC_Dataset_ped1-12685_time1560-1620_interp9_xrange5-25_yrange15-35`: Refer to the basic scenario.
  -   `simulation: Simulate the basic scenario, where the number of pedestrians and their appearance time, initial position, and destination are the same as in the basic scenario.

For each `*.npy` file, the data is stored in the following format:
```python
(metadata, trajectories, destinations, obstacles) = np.load(file_path, allow_pickle=True)

```
And here is the description of each element:
- **metadata**: A dictionary that contains the metadata of the dataset.
- **trajectories**: A list of trajectories. For each trajectory, `np.array(trajectory)` is a `np.array` with shape `(T, 3)`, where `T` is the number of time steps where the pedestrian appears in the scenario, and each row is the x-position, y-position and time of that pedestrian.
- **destinations**: `np.array(destinations)` is an `np.array` with shape `(N, 1, 3)`, where `N` is the number of pedestrians. `np.array(destinations)[:, 0, :2]` is the xy-position of the destination, while `np.array(destinations)[:, 0, 2]` is the time they reach the destination.
- **obstacles**: An `np.array` with shape `(M, 2)`, where `M` is the number of obstacle points, and each row is the xy-position of the obstacle.
