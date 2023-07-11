import os
import subprocess
from dataclasses import dataclass, field
from typing import List
import warnings
import re


def allresources():
    #get job IDs
    shell_run = subprocess.run("qstat -u $USER | cut -c1-8 | grep -Eo '[0-9]{1,7}' | tr '\n' ' '",
                                shell=True,
                                capture_output=True,
                                encoding="utf-8",
                                )
    if shell_run.stderr: raise RuntimeError(shell_run.stderr)
    ids = shell_run.stdout.split()

    resources_dict = {}
    for id in ids:
        shell_run = subprocess.run(f'''qstat -f {id} | grep -e resources_used -e Job_Name -e Resource_List.mem -e Resource_List.ncpus''',
                                   shell=True,
                                   capture_output=True,
                                   encoding="utf-8")
        job_resources = resources_out_to_dict(shell_run.stdout)
        resources_dict[id] = job_resources
    return resources_dict

def to_gb(s):
    kb_used = int(s[:-2])
    return round(kb_used / 1e6, 2)

def resources_out_to_dict(shell_out):
    job_resources = [s.lstrip() for s in shell_out.split('\n') if s]
    resources_dict = {}
    for line in job_resources:
        components = line.split()
        resources_dict[components[0]] = components[-1]

    if 'resources_used.mem' in resources_dict.keys():
        # then vmem should be there too
        resources_dict['resources_used.mem'] = to_gb(resources_dict['resources_used.mem'])
        resources_dict['resources_used.vmem'] = to_gb(resources_dict['resources_used.vmem'])

    return resources_dict
        

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
    job_script: List[str] = field(default_factory=lambda: ['export TMPDIR=/glade/scratch/$USER/temp',
                                                           'mkdir -p $TMPDIR',
                                                           'source /etc/profile.d/modules.sh'])

    def set_job_attrs(self, **kwargs):
        keys = list(kwargs.keys())
        class_keys = list(self.__dict__.keys())
        for k in keys:
            if k not in class_keys: raise AttributeError(f'{k} is not an attr and cannot be set')

        warning_keys = []
        for k in keys:
            if k in ['account', 'user_list', 'queue', 'mail_option']:
                warning_keys.append(k)
        if warning_keys:
            warnings.warn(f"You are setting protected attrs: {warning_keys}")
        
        self.__dict__.update(kwargs)

    def set_job_script(self, commands: List[str]):
        self.job_script = commands
    
    def append_job_script(self, commands: List[str]):
        self.job_script += commands
    
    def submit(self, commands: List[str] = []):
        #takes current job_script and appends commands to it, submits PBS job, prints jobid
        lines = self._args_to_lines() + self.job_script + commands
        
        temp_qsub_file = 'qsub_temp_script'
        with open(temp_qsub_file, 'w') as f: # note: this overwrites the previous file
            for line in lines:
                f.write(f"{line}\n")
        #submit job
        shell_run = subprocess.run(f"qsub {temp_qsub_file}",
                                shell=True,
                                capture_output=True,
                                encoding="utf-8",
                                )
        # output jobid or error message
        jobID = shell_run.stdout.rstrip()
        stderr = shell_run.stderr.rstrip()
        if stderr: 
            raise ChildProcessError(f'Error: {stderr}.\n Please see qsub script {temp_qsub_file}') # doesn't delete the qsub script
        elif not jobID: 
            warnings.warn(f"No jobID returned, job may not have submitted. please check qstat. use makefile() method to diagnose problems")
        else:
            print(jobID)

        #remove qsub script file
        os.remove(temp_qsub_file)
        return jobID
    
    def makefile(self, commands=[], filename='qsub_script_out'):
        lines = self._args_to_lines() + self.job_script + commands
        temp_qsub_file = filename
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




