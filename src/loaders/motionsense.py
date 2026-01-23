import numpy as np
import pandas as pd
import zipfile
from pathlib import Path
import torch
from torch.utils.data import Subset, DataLoader, TensorDataset, Dataset

ACT_LABELS = ["dws","ups", "wlk", "jog", "std", "sit"]
SDT = ["attitude", "gravity", "rotationRate", "userAcceleration"]
GENS = ['male', 'female']
TRIAL_CODES = {
        ACT_LABELS[0]: [1,2,11],
        ACT_LABELS[1]: [3,4,12],
        ACT_LABELS[2]: [7,8,15],
        ACT_LABELS[3]: [9,16],
        ACT_LABELS[4]: [6,14],
        ACT_LABELS[5]: [5,13],
}

DATA_DIR = "data/MotionSense"
SETS = ["train", "valid", "test"]

class MSLoader:
    def __init__(self, batch_size: int, num_workers: int):
        self.name = 'MS'
        self.data_dir = Path(DATA_DIR)

        if not self.data_dir.exists():
            raise RuntimeError(
                    f"The dir {self.data_dir} doesn't exist. "
                    "Please check the ``root`` path or run `dvc pull` to download it"
                    )
        # Extract zip files if not already extracted
        zips_to_extract = [f for f in self.data_dir.glob('*.zip') if not (self.data_dir / f.stem).exists()]
        for zip_file in zips_to_extract:
            zf = zipfile.ZipFile(zip_file, 'r')
            zf.extractall(self.data_dir)
            zf.close()

        self.data = self.get_data()

        whole_tset = MSSubset(self.data, "train")
        testset  = MSSubset(self.data, "test", whole_tset.mean, whole_tset.std)

        if not (self.data_dir/"perm.pt").exists(): # For a fixed valid set every run (reporducibility)
            perm = torch.randperm(len(whole_tset))
            torch.save(perm, self.data_dir/"perm.pt")
        perm = torch.load(self.data_dir/"perm.pt")

        val_len = int(len(perm)*0.1) # 10% for validation

        trainset = Subset(whole_tset, perm[val_len:])
        validset = Subset(whole_tset, perm[:val_len])

        tensor_missing = sum([(not (self.data_dir/f"{dset}_tensors.pt").exists() 
                               or not (self.data_dir/f"{dset}_labels.pt").exists())
                              for dset in SETS])
        if tensor_missing:
               self.get_tensors([trainset, validset, testset])

        train_tensors, train_labels = (torch.load(self.data_dir/"train_tensors.pt"), 
                                       torch.load(self.data_dir/"train_labels.pt"))
        valid_tensors, valid_labels = (torch.load(self.data_dir/"valid_tensors.pt"), 
                                       torch.load(self.data_dir/"valid_labels.pt"))
        test_tensors, test_labels = (torch.load(self.data_dir/"test_tensors.pt"), 
                                       torch.load(self.data_dir/"test_labels.pt"))

        trains = TensorDataset(train_tensors, train_labels)
        valids = TensorDataset(valid_tensors, valid_labels)
        tests = TensorDataset(test_tensors, test_labels)

        self.train = DataLoader(trains, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers)
        self.valid = DataLoader(valids, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers)
        self.test = DataLoader(tests, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers)

        self.sample_rate = 50  # Hz
        self.n_datapoints = 12

        self.in_chan = 1
        self.in_size = (1, self.sample_rate * self.n_datapoints)
        self.out_dim = len(ACT_LABELS)

    def get_tensors(self, sets):
        for i, dset in enumerate(SETS):
            images, labels = [], []
            for img, label in sets[i]:
                images.append(img)
                labels.append(torch.tensor(label))
            torch.save(torch.stack(images), self.data_dir/f"{dset}_tensors.pt") 
            torch.save(torch.stack(labels), self.data_dir/f"{dset}_labels.pt")

    def get_config(self):
        return {
                "task": self.name,
                "sample_rate": self.sample_rate,
                "n_datapoints": self.n_datapoints,
                "in_chan": self.in_chan,
                "in_size": self.in_size,
                "out_dim": self.out_dim,
                "train_samples": len(self.train.dataset),
                "valid_samples": len(self.valid.dataset),
                "test_samples": len(self.test.dataset),
                }

    def get_ds_infos(self):
        """
        Read the file includes data subject information.
        
        Data Columns:
        0: code [1-24]
        1: weight [kg]
        2: height [cm]
        3: age [years]
        4: gender [0:Female, 1:Male]
        
        Returns:
            A pandas DataFrame that contains information about data subjects' attributes 
        """ 

        return pd.read_csv(self.data_dir/"data_subjects_info.csv")

    def set_data_types(self, data_types: list[str]):
        """
        Select the sensors and the mode to shape the final dataset.
        
        Args:
            data_types: A list of sensor data type from this list: [attitude, gravity, rotationRate, userAcceleration] 

        Returns:
            It returns a list of columns to use for creating time-series from files.
        """
        dt_list = []
        for t in data_types:
            if t != "attitude":
                dt_list.append([t+".x",t+".y",t+".z"])
            else:
                dt_list.append([t+".roll", t+".pitch", t+".yaw"])

        return dt_list

    def create_time_series(self, dt_list, act_labels, trial_codes, mode="mag", 
                           labeled=True):
        """
        Args:
            dt_list: A list of columns that shows the type of data we want.
            act_labels: list of activites
            trial_codes: list of trials
            mode: It can be "raw" which means you want raw data
            for every dimention of each data type,
            [attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)].
            or it can be "mag" which means you only want the magnitude for each data type: (x^2+y^2+z^2)^(1/2)
            labeled: True, if we want a labeld dataset. False, if we only want sensor values.

        Returns:
            It returns a time-series of sensor data.
        
        """
        num_data_cols = len(dt_list) if mode == "mag" else len(dt_list*3)

        if labeled:
            dataset = np.zeros((0,num_data_cols+7)) # "7" --> [act, code, weight, height, age, gender, trial] 
        else:
            dataset = np.zeros((0,num_data_cols))
            
        ds_list = self.get_ds_infos()
        
        for sub_id in ds_list["code"]:
            for act_id, act in enumerate(act_labels):
                for trial in trial_codes[act_id]:
                    fname = 'A_DeviceMotion_data/'+act+'_'+str(trial)+'/sub_'+str(int(sub_id))+'.csv'
                    raw_data = pd.read_csv(self.data_dir/fname)
                    raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                    vals = np.zeros((len(raw_data), num_data_cols))
                    for x_id, axes in enumerate(dt_list):
                        if mode == "mag":
                            vals[:,x_id] = (raw_data[axes]**2).sum(axis=1)**0.5        
                        else:
                            vals[:,x_id*3:(x_id+1)*3] = raw_data[axes].values
                        vals = vals[:,:num_data_cols]
                    if labeled:
                        lbls = np.array([[act_id,
                                sub_id-1,
                                ds_list["weight"][sub_id-1],
                                ds_list["height"][sub_id-1],
                                ds_list["age"][sub_id-1],
                                ds_list["gender"][sub_id-1],
                                trial          
                               ]]*len(raw_data))
                        vals = np.concatenate((vals, lbls), axis=1)
                    dataset = np.append(dataset,vals, axis=0)
        cols = []
        for axes in dt_list:
            if mode == "raw":
                cols += axes
            else:
                cols += [str(axes[0][:-2])]
                
        if labeled:
            cols += ["act", "id", "weight", "height", "age", "gender", "trial"]
        
        dataset = pd.DataFrame(data=dataset, columns=cols)
        return dataset

    def get_data(self):
        trial_codes = [TRIAL_CODES[act] for act in ACT_LABELS]
        dt_list = self.set_data_types(SDT)
        dataset = self.create_time_series(dt_list, ACT_LABELS, trial_codes, mode="raw", 
                                    labeled=True)
        return dataset

class MSSubset(Dataset):
    def __init__(self, data: Path, fold="train", mean=None, std=None):
        data = (data[data['trial'] <= 10].values if fold == "train" 
                else data[data['trial'] > 10].values)
        self.data, self.act_labels, self.gen_labels, self.mean, self.std = \
                self.time_series_to_section(data, len(ACT_LABELS), len(GENS)-1, 
                                            sliding_window_size=50, 
                                            step_size_of_sliding_window=10, 
                                            standardize=True, mean=mean, std=std)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.act_labels[index][0]

    def time_series_to_section(self, dataset, num_act_labels, num_gen_labels, 
                               sliding_window_size, step_size_of_sliding_window, 
                               standardize = False, mean = False, std = False):
        data = dataset[: , 0:-(num_act_labels+num_gen_labels)]
        act_labels = dataset[: , -(num_act_labels+num_gen_labels):-(num_gen_labels)]
        gen_labels = dataset[: , -(num_gen_labels)]
        mean = 0
        std = 1
        
        if standardize:
            ## Standardize each sensor’s data to have a zero mean and unity standard deviation.
            ## As usual, we normalize test dataset by training dataset's parameters 
            if mean == None:
                mean = data.mean(axis=0)
                std = data.std(axis=0)
            data -= mean
            data /= std

        ## We want the Rows of matrices show each Feature and the Columns show time points.
        data = data.T
                
        size_features = data.shape[0]
        size_data = data.shape[1]
        number_of_secs = round(((size_data - sliding_window_size)/step_size_of_sliding_window))
                
        ##  Create a 3D matrix for Storing Snapshots  
        secs_data = np.zeros((number_of_secs , size_features , sliding_window_size ))
        act_secs_labels = np.zeros((number_of_secs, num_act_labels))
        gen_secs_labels = np.zeros(number_of_secs)
        
        k=0    
        for i in range(0, (size_data)-sliding_window_size, step_size_of_sliding_window):
            j = i // step_size_of_sliding_window
            if(j>=number_of_secs):
                break
            if(gen_labels[i] != gen_labels[i+sliding_window_size-1]): 
                continue
            if(not (act_labels[i] == act_labels[i+sliding_window_size-1]).all()): 
                continue    
            secs_data[k] = data[0:size_features, i:i+sliding_window_size]
            act_secs_labels[k] = act_labels[i].astype(int)
            gen_secs_labels[k] = gen_labels[i].astype(int)
            k = k+1
        secs_data = torch.tensor(secs_data[0:k], dtype=torch.float)
        act_secs_labels = torch.tensor(act_secs_labels[0:k], dtype=torch.long)
        gen_secs_labels = torch.tensor(gen_secs_labels[0:k], dtype=torch.long)
        
        return secs_data, act_secs_labels, gen_secs_labels, mean, std

    def get_train_data(self):
        sliding_window_size = 50 # 50 Equals to 1 second for MotionSense Dataset (it is on 50Hz samplig rate)
        step_size_of_sliding_window = 10 
        train_data, act_labels, gen_labels, mean, std = \
                time_series_to_section(self.data.copy(), num_act_labels, num_gen_labels, 
                                       sliding_window_size, step_size_of_sliding_window,
                                       standardize = True)

        test_data, act_test_labels, gen_test_labels, test_mean, test_std = \
                time_series_to_section(test_ts.copy(), num_act_labels, num_gen_labels,
                                       sliding_window_size, step_size_of_sliding_window,
                                       standardize = True, 
                                       mean = train_mean, std = train_std)

    def get_test_data(self, mean, std):
        data, act_labels, gen_labels, mean, std = time_series_to_section(
                self.data.copy(), num_act_labels, num_gen_labels, sliding_window_size=50, 
                step_size_of_sliding_window=10, standardize=True, mean=mean, std=std)
        return data, act_labels, gen_labels, mean, std
