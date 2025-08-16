import os
import warnings
import numpy as np
import h5py
import mat73
from scipy.io import loadmat

from src.utils import helper_functions as helper
from src.data.data_processing import ProcessData
from src.utils import pre_processing_functions as pre_process

from pathlib import Path
from IPython.display import display, HTML
from typing import Union, Dict, Optional
import glob as glob

class LoadData:
    def __init__(self, dataset_name: str):
        """
        Initialize data loader for specified dataset.
        
        Args:
            dataset_name (str): Name of the dataset (e.g., 'kinsky')
        """
        self.dataset_name = dataset_name
        self.data_path = self._resolve_data_path()
        
        
        if not os.path.exists(self.data_path):
            warnings.warn(f"Dataset '{dataset_name}' not found at {self.data_path}")
        else:
            print(f"Dataset path resolved: {self.data_path}")


    def data_lakes(self) -> str:
        """ Just a placeholder for data lakes URL"""
        'https://www.cell.com/cell-reports/fulltext/S2211-1247(25)00162-7?uuid=uuid%3A90314a59-1e6c-4ba4-9584-1c6aab11b469'
        
        return None
    


    def _resolve_data_path(self) -> Path:
        """Resolve the absolute path to the dataset directory."""
        current_file = Path(__file__).absolute()
        project_root = current_file.parent.parent.parent  # Adjust based on your structure
        return project_root / 'data' / self.dataset_name


    def load_etter(self,experiment):
        """
        Load Etter dataset for specific experiment.
        
        Args:
            experiment (str): linear_track for 1D or open_field for 2D experiments
            
        Returns:
            Dictionary containing all data components
        """
        mat_file = f'{self.data_path}/{experiment}/behav_time.mat'
        mat_dict = loadmat(mat_file, simplify_cells=True)
        behav_time = mat_dict['behav_time']


        mat_file = f'{self.data_path}/{experiment}/behav_vec.mat'
        mat_dict = loadmat(mat_file, simplify_cells=True)
        behav_vec = mat_dict['behav_vec']

        mat_file = f'{self.data_path}/{experiment}/ca_data.mat'
        mat_dict = loadmat(mat_file, simplify_cells=True)
        ca_data = mat_dict['ca_data'].T
        
        mat_file = f'{self.data_path}/{experiment}/ca_time.mat'
        mat_dict = loadmat(mat_file, simplify_cells=True)
        ca_time = mat_dict['ca_time']
        
        mat_file = f'{self.data_path}/{experiment}/ca_trace.mat'
        mat_dict = loadmat(mat_file, simplify_cells=True)
        ca_trace = mat_dict['ca_trace'] # this is just one selected cell they used in their figures, let`s ignore it

        raw = ProcessData.sync_imaging_to_video(ca_data, ca_time, behav_time)
        sampling_rate = 1/np.nanmedian(np.diff(behav_time))

        signal_type = 'filtered'
        filtered_signal = pre_process.preprocess_signal(raw, sampling_rate, signal_type, z_threshold = 2)

        signal_type = 'diff'
        diff_signal = pre_process.preprocess_signal(raw, sampling_rate, signal_type, z_threshold = 2)
        
        signal_type = 'binary'
        binary_signal = pre_process.preprocess_signal(raw, sampling_rate, signal_type, z_threshold = 2)

        x_coordinates = behav_vec if experiment == 'linear_track' else behav_vec[:,0]
        y_coordinates = None if experiment == 'linear_track' else behav_vec[:,1]
        speed, smoothed_speed = helper.get_speed(x_coordinates, y_coordinates, behav_time, speed_smoothing_sigma=1)

        if experiment == 'linear_track':
            environment_edges = [[0,100],[]]
        elif experiment == 'open_field':
            environment_edges = [[0,50],[0,45]]
            
        return {
            'position': {
                'x': x_coordinates,
                'y': y_coordinates,
                'time': behav_time
            },
            'traces': {
                'raw': raw,
                'filtered': filtered_signal,
                'diff': diff_signal,
                'binary': binary_signal
            },
            
            'speed': speed,
            'sampling_rate': sampling_rate,
            'environment_edges': environment_edges
        }

    def load_hanna(self):

        filename = f'{self.data_path}/real_signal.npy'
        data = np.load(filename,allow_pickle = True).item()


        return {
            'position': {
                'x': data['x_coordinates'],
                'y': data['y_coordinates'],
                'time': data['time_vector']
            },
            'traces': {
                'raw': data['signal'],
                'filtered': [],
                'diff': [],
                'binary': []
            },
            'speed': [],
            'sampling_rate': data['sampling_rate'],
            'environment_edges': data['environment_edges']
        }

        

    def load_kinsky(self, mouse_id: int, day: int, session: int) -> Dict[str, np.ndarray]:
        """
        Load Kinsky dataset for specific mouse, day, and session.
        
        Args:
            mouse_id (int): Mouse ID (1-4)
            day (int): Day number (1-8)
            session (int): Session number (1-2)
            
        Returns:
            Dictionary containing all data components
        """
        
        data_dir = self.data_path / f'mouse_{mouse_id}' / f'day_{day}' / f'session_{session}'
        mat_file = data_dir / 'Pos_align.mat'

        
        if not mat_file.exists():
            raise FileNotFoundError(f"MAT file not found at {mat_file}")
            
        try:
            mat_dict = loadmat(mat_file, simplify_cells=True)
        except Exception as e:
            raise ValueError(f"Error loading MAT file: {e}")

        return {
            'position': {
                'x': mat_dict['x_adj_cm'],
                'y': mat_dict['y_adj_cm'],
                'time': mat_dict['time_interp']
            },
            'traces': {
                'raw': mat_dict['RawTrace'],
                'filtered': mat_dict['LPtrace'],
                'diff': mat_dict['DFDTtrace'],
                'binary': mat_dict['PSAbool']
            },
            'speed': mat_dict['speed'],
            'sampling_rate': 20,  # Hz, from paper
            'environment_edges': []
        }



    def load(self, **kwargs) -> Union[Dict, None]:
        """
        Main loading interface that routes to dataset-specific loaders.
        
        Args:
            **kwargs: Parameters specific to each dataset
                      For 'kinsky': mouse_id, day, session
                      
        Returns:
            Loaded data or None if dataset not found
        """
        if self.dataset_name == 'kinsky':
            return self.load_kinsky(**kwargs)
        
        elif self.dataset_name == 'hanna':
            return self.load_hanna()

        elif self.dataset_name == 'etter':
            return self.load_etter(**kwargs)
        
        else:
            warnings.warn(f"No loader implemented for dataset: {self.dataset_name}")
            return None
        


    def list_available(self) -> Dict[str, Union[Dict[str, list], list]]:
        """List available mice with their actual days and sessions."""
        match self.dataset_name:
            case 'kinsky':
                available = {}
                for mouse_dir in self.data_path.glob('mouse_*'):
                    mouse_id = mouse_dir.name
                    available[mouse_id] = {
                        'days': [],
                        'sessions': []
                    }
                    
                    # Scan for available days
                    for day_dir in mouse_dir.glob('day_*'):
                        day_num = int(day_dir.name.split('_')[1])
                        available[mouse_id]['days'].append(day_num)
                        
                        # Scan for available sessions in each day
                        sessions = set()
                        for session_dir in day_dir.glob('session_*'):
                            session_num = int(session_dir.name.split('_')[1])
                            sessions.add(session_num)
                        available[mouse_id]['sessions'] = sorted(sessions)
                
                return {'available': available}
                
            case _:
                return {}






    def show_jupyter_tree(self):
        """Display retro-style folder tree in JupyterLab"""
        html = ["<div style='font-family: monospace; white-space: pre;'>"]

        match self.dataset_name:
            case 'kinsky':
                html.append("Dataset: kinsky")
                for mouse in sorted(self.data_path.glob('mouse_*')):
                    mouse_id = mouse.name.split('_')[1]
                    html.append(f"+ Mouse {mouse_id}")
                    
                    for day in sorted(mouse.glob('day_*')):
                        day_num = day.name.split('_')[1]
                        html.append(f"|  + Day {day_num}")
                        
                        sessions = list(sorted(day.glob('session_*')))
                        if sessions:
                            for session in sessions:
                                sess_num = session.name.split('_')[1]
                                html.append(f"|  |  + Session {sess_num}")
                                
                                files = list(sorted(session.glob('*')))
                                if files:
                                    for f in files:
                                        if f.is_file():
                                            html.append(f"|  |  |  - {f.name}")
                                else:
                                    html.append(f"|  |  |  (no files)")
                        else:
                            html.append(f"|  |  (no sessions)")
            case _:
                html.append("No tree available for this dataset type")
        
        html.append("</div>")
        display(HTML('\n'.join(html)))
