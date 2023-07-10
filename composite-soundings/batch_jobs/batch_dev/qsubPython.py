import os
import subprocess
from dataclasses import dataclass, field
from typing import List
import warnings

@dataclass
class qsubPython:
    account: str
    user_list: str
    queue: str
    mail_option: str = 'a'
    jobname: str = None
    walltime: str = '01:00:00'
    resources: str = None
    outfile: str = 'out.out'
    errorfile: str = 'out.out'
    job_script: List[str] = field(default_factory=list)

    def set_job_args(self, **kwargs):
        keys = list(kwargs.keys())
        class_keys = list(self.__dict__.keys())

        for k in keys:
            if k not in class_keys: raise ValueError(f'{k} is not an attr and cannot be set')

        warning_keys = []
        for k in keys:
            if k in ['account', 'user_list', 'queue', 'mail_option']:
                warning_keys.append(k)
        if warning_keys:
            warnings.warn(f"You are setting protected keys: {warning_keys}")
        
        self.__dict__.update(kwargs)

    def set_job_script(self, commands: List[str]):
        self.job_script = commands
    
    def append_job_script(self, commands: List[str]):
        self.job_script += commands
    
    def submit(self, commands: List[str] = []):
        #takes current job_script and appends commands to it, submits PBS job, prints jobid
        job_script = self.job_script + commands
        if not job_script: raise ValueError("no lines to run in body of script!")

        lines = self._args_to_lines() + job_script

        temp_qsub_file = 'qsub_temp_script'
        with open(temp_qsub_file, 'w') as f:
            for line in lines:
                f.write(f"{line}\n")
        #submit job
        shell_run = subprocess.run(f"qsub {temp_qsub_file}",
                                shell=True,
                                capture_output=True,
                                encoding="utf-8",
                                )
        # output jobid or error message
        jobID = shell_run.stdout
        stderr = shell_run.stderr
        if stderr: 
            raise ValueError(f'Error: {stderr}')
        elif not jobID: warnings.warn(f"jobID is {jobID}, job may not have submitted. please check qstat. use makefile() method to diagnose problems")
        print(jobID)

        #remove qsub script file
        os.remove(temp_qsub_file)
        return jobID
    def makefile(self, commands=[]):
        job_script = self.job_script + commands
        if not job_script: raise ValueError("no lines to run in body of script!")

        lines = self._args_to_lines() + job_script

        temp_qsub_file = 'qsub_temp_script'
        with open(temp_qsub_file, 'w') as f:
            for line in lines:
                f.write(f"{line}\n")

    def _args_to_lines(self):
        empty = []
        for attr, value in vars(self).items():
            if not value: empty.append(attr)
        if empty:
            raise ValueError(f'{empty} are unset')

        job_config = [f'-A {self.account}',
                      f'-M {self.user_list}',
                      f'-q {self.queue}',
                      f'-m {self.mail_option}',
                      f'-N {self.jobname}',
                      f'-l walltime={self.walltime}',
                      f'-l {self.resources}',
                      f'-o {self.outfile}',
                      f'-e {self.errorfile}']
        job_config = [f'#PBS {s}' for s in job_config]
        return ['#!/bin/bash -l'] + job_config





