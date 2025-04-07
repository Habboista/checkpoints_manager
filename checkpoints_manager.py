import os
import torch

class CheckpointsManager:
    """Class for automatically loading and saving checkpoints of
    specified modules in an organized directory tree.

    Structure of the directory tree containing the checkpoints:

        - experiment_dir
            |
            | - run_00
                |
                |- checkpoints
                    |
                    |- model
                        |
                        |- 00.pth
                        |- 01.pth
                        |- ...
                    |
                    |- scheduler
                        |
                        |- 00.pth
                        |- ...
                    |
                    |- optimizer
                        |
                        |-...
            |
            | - run_01
                |
                |- checkpoints
                    |
                    |- ...
    """

    def __init__(self,
        modules_to_track,
        experiment_dir_path,
        epochs_per_run,
        verbose=True
    ):
        # Passed arguments
        self.modules_to_track = modules_to_track
        self.experiment_dir_path = experiment_dir_path
        self.epochs_per_run = epochs_per_run
        self.verbose = verbose
        
        os.makedirs(self.experiment_dir_path, exist_ok=True)

        # Epoch from which to resume or next epoch to compute,
        # if it is 0, the training is yet to start
        self.start_epoch = None

        # Path of the run directory where to store checkpoints
        # of the current ongoing training run
        self.current_run_dir_path = None

        self.scan_experiment_dir()

    
    def load_last_checkpoint(self): 
        """Load state dict of objects specified by the dictionary
        modules_to_track of last run/epoch.
        E.g. modules_to_track = {
            'model': model,
            'optmizer': optimizer,
            'scheduler': scheduler,
        }
        If last run completedd, create folder for checkpoints of new run.
        """
        self.scan_experiment_dir()                 

        # No checkpoints to load
        if self.start_epoch == 0:
            return
        
        # Otherwise, load checkpoints
        for module_name, module in self.modules_to_track.items():
            checkpoint_module_path = \
                self.get_checkpoint_module_path(
                    module_name,
                    self.start_epoch - 1,
                )
            module.load_state_dict(
                torch.load(checkpoint_module_path, map_location='cpu')
            )
            if self.verbose:
                print(f"Loaded checkpoint {self.start_epoch - 1:02d}" +
                    f" / {self.epochs_per_run - 1:02d}" +
                    f"for module {module_name}.")


    def save_checkpoint(self): 
        """Save state dict of objects specified in modules_to_track. 
        of run/epoch. It automatically keeps track of current epoch.
        """
        
        for module_name, module in self.modules_to_track.items():
            checkpoint_module_path = \
                self.get_checkpoint_module_path(
                    module_name,
                    self.start_epoch,
                )
            torch.save(
                module.state_dict(),
                checkpoint_module_path,
            )
            if self.verbose:
                print(f"Saved checkpoint {self.start_epoch:02d}" +
                    f" / {self.epochs_per_run - 1:02d}" +
                    f"for module {module_name}.")

        self.start_epoch += 1
        if self.start_epoch == self.epochs_per_run:
            # Start next run
            current_run_index = \
                self.get_run_index_from_path(self.current_run_dir_path)
            if self.verbose:
                print(f"Completed run_{current_run_index:02d}.")
            self.make_run_dir(current_run_index + 1)
        

    def get_checkpoint_module_path(self, name, epoch):
        """Return the path of the checkpoint module of given name.
        For instance, name='model', epoch=3 will return a path like:
            '.../checkpoints/model/03.pth'.
        """
        return os.path.join(
            self.current_run_dir_path,
            name,
            f'{epoch:02d}.pth',
        )


    def get_number_of_runs(self):
        run_dir_list = os.listdir(self.experiment_dir_path)
        return len(run_dir_list)


    def get_start_epoch(self):
        return self.start_epoch
    
    
    def get_current_run_dir_path(slef):
        return self.current_run_dir_path


    def get_run_index_from_path(self, run_dir_path):
        """From ".../run_idx" return idx as an integer"""
        return int(os.path.basename(run_dir_path).split('_')[-1])
    
    
    def make_run_dir(self, index):
        """Create directory for new run of given index inside
        the directory specified by self.experiment_dir_path.
        It also updates internal variables used for loading
        checkpoints.
        """
        run_dir_path = os.path.join(
            self.experiment_dir_path,
            f'run_{index:02d}',
        )
        
        # Run dir
        os.makedirs(
            run_dir_path,
        )
        
        # Checkpoint dirs
        for module_name in self.modules_to_track:
            os.makedirs(
                os.path.join(
                    run_dir_path,
                    module_name,
                ),
            )
        
        self.start_epoch = 0
        self.current_run_dir_path = run_dir_path
        if self.verbose:
            current_run_dir = \
                os.path.basename(self.current_run_dir_path)
            print(f'Starting {current_run_dir}.')


    def scan_experiment_dir(self):
        """Scan the content of the experiment_dir.
        It checks which runs are completed and from which epoch
        to resume training. It also updates internal variable of
        the class, used for loading checkpoints.
        """
        # Are there no runs yet?
        num_runs = self.get_number_of_runs()
        if num_runs == 0:

            # Create the first run
            if self.verbose:
                print('No runs found to load from.')
            self.make_run_dir(0)
            return

        # There is at least one run, pick the most recent
        last_run_dir_path = self.get_last_run_dir_path()

        # Is the latest run completed?
        if self.is_run_completed(last_run_dir_path):

            # Yes, make new run
            if self.verbose:
                last_run_dir = os.path.basename(last_run_dir_path)
                print(f'Found {last_run_dir} completedd.')
            current_run_index = \
                self.get_run_index_from_path(last_run_dir_path) + 1
            self.make_run_dir(current_run_index)
        else:
            
            # No, get last checkpoint in current run.
            self.current_run_dir_path = last_run_dir_path
            self.start_epoch = \
                self.get_last_epoch_index(self.current_run_dir_path) + 1


    def get_last_run_dir_path(self):
        """Assuming there is at least one run directory in
        self.experiment_dir_path, return the one with
        highest index. E.g.:
        
        - experiment_dir
            |
            |- run_00
            |- run_01
            |- run_02

        It is returned ".../experiment_dir/run_02".
        """ 
        run_dir_list = sorted(os.listdir(self.experiment_dir_path))
        last_run_dir = run_dir_list[-1]

        last_run_dir_path = os.path.join(
            self.experiment_dir_path,
            last_run_dir,
        )

        return last_run_dir_path


    def get_last_epoch_index(self, run_dir_path):
        model_dir_path = os.path.join(run_dir_path, 'model')
        optimizer_dir_path = os.path.join(run_dir_path, 'optimizer')
        scheduler_dir_path = os.path.join(run_dir_path, 'scheduler')
        
        return min(
            self._get_last_epoch_index(model_dir_path),
            self._get_last_epoch_index(optimizer_dir_path),
            self._get_last_epoch_index(scheduler_dir_path),
        )


    def _get_last_epoch_index(self, dir_path):
        """ Returns the index of the last epoch for which a checkpoint
        is available in a folder of the following type:

        - dir
            |- 00.pth
            |- 01.pth
            |- ...

        If the specified folder is empty, -1 is returned.
        """
        if not os.path.isdir(dir_path):
            raise NotADirectoryError

        checkpoints = sorted(os.listdir(dir_path))

        if len(checkpoints) == 0:
            return -1

        last_epoch = int(checkpoints[-1].split('.')[0])
        return last_epoch


    def is_run_completed(self, run_dir_path):
        """Check whether model, optimizer and scheduler corresponding to
        final epoch are saved in the run directory."""
        if self.is_checkpoint_dir_full(run_dir_path):
            return True
        else:
            return False
    

    def is_run_corrupted(self, run_dir_path):
        raise NotImplementedError


    def is_checkpoint_dir_full(self, checkpoints_dir_path):
        last_epoch = self.get_last_epoch_index(checkpoints_dir_path)
        return last_epoch == self.epochs_per_run - 1
