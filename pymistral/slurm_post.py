import os
from subprocess import Popen


def setup_slurm(fh, job_name, tasks_per_node=1, partition='shared', time='2:00:00'):
    fh.write("#SBATCH --job-name=%s.job\n" % job_name)
    fh.write("#SBATCH --output=%s.log\n" % job_name)
    fh.write("#SBATCH --error=%s.log\n" % job_name)
    fh.write("#SBATCH --partition="+partition+"\n")
    fh.write("#SBATCH --tasks-per-node="+str(tasks_per_node)+"\n")
    fh.write("#SBATCH --time="+time+"\n")
    fh.write("#SBATCH --mail-type=FAIL\n")
    fh.write("#SBATCH --mail-user=aaron.spring@mpimet.mpg.de\n")
    fh.write("#SBATCH --account=mh0727\n")
    #fh.write("#SBATCH --propagate=STACK,CORE\n")
    #fh.write("ulimit -s 390625 # * 1024 B = 400 MB\n")


def setup_my_jupyter(fh):
    fh.write("PATH=$PATH:/work/mh0727/m300524/anaconda3/bin\n")
    fh.write("export PATH\n")
    fh.write("source activate my_jupyter\n")

# path = '~/'+os.getcwd()[30:]
path = os.getcwd()
# path = '$HOME/'+os.getcwd()[30:]
def send_to_slurm(job_py_str, job_name_str, path=path, test=False, tasks_per_node=1, partition='shared', time='2:00:00'):
    """Send python to SLURM.

    Example:
    *   send_to_slurm(create_job_python_file(job_name, varnames,
        enslist=enslist, model='mpiom',outdatatype='data_2d_mm', ending='.nc',
        curv=True), job_name, test=False)

    Args:
        job_py_str (type): Description of parameter `job_py_str`.
        job_name_str (type): Description of parameter `job_name_str`.
        path (str): path of user.
        test (type): Description of parameter `test`. Defaults to False.

    Returns:
        type: Description of returned object.

    """
    os.chdir(path)
    job_directory = ''
    job_file_str = os.path.join(job_directory, "%s.job" % job_name_str)
    if os.path.exists(job_file_str):
        os.remove(job_file_str)
    with open(job_file_str, "w") as fh:
        fh.write("#!/bin/bash\n")
        setup_slurm(fh, job_name_str, tasks_per_node=tasks_per_node, partition=partition, time=time)
        # load conda env
        setup_my_jupyter(fh)
        # change to folder
        write(fh, 'cd '+path)
        # run python
        fh.write("python %s \n" % job_py_str)

    if test:
        os.system("cat %s" % job_file_str)
    else:
        print('sending!')
        print("sbatch", job_file_str)
        # os.system("sbatch %s " % job_file_str)
        process = Popen(['sbatch', job_file_str])



def setup_PM_postprocess(fh):
    write(fh, "from PMMPIESM.setup import postprocess_PM")


def setup_GE_postprocess(fh):
    write(fh, "from PMMPIESM.setup import postprocess_GE")


def write(fh, line):
    fh.write(line + "\n")


def eval_args(varnames, **kwargs):
    """Convert arguments postprocess_PM to str.

    Args:
        varnames (type): Description of parameter `varnames`.
        **kwargs (type): Description of parameter `**kwargs`.

    Returns:
        args_str (str): Description of returned object.

    """
    args_str = str(varnames)
    for key, value in kwargs.items():
        if isinstance(value, str):
            args_str += ",\n\t" + key + "='" + str(value) + "'"
        else:
            args_str += ",\n\t" + key + "=" + str(value)
    return args_str


def eval_args_str(varnames, **kwargs):
    """Convert arguments postprocess_PM to str.

    Args:
        varnames (type): Description of parameter `varnames`.
        **kwargs (type): Description of parameter `**kwargs`.

    Returns:
        args_str (str): Description of returned object.

    """
    args_str = "'" + varnames + "'"
    for key, value in kwargs.items():
        if isinstance(value, str):
            args_str += ",\n\t" + key + "='" + str(value) + "'"
        else:
            args_str += ",\n\t" + key + "=" + str(value)
    return args_str


def eval_args_str_name(varname, **kwargs):
    """Convert arguments postprocess_PM to str.

    Args:
        varnames (type): Description of parameter `varnames`.
        **kwargs (type): Description of parameter `**kwargs`.

    Returns:
        args_str (str): Description of returned object.

    """
    args_str = "varname='" + varname + "'"
    for key, value in kwargs.items():
        if isinstance(value, str):
            args_str += ",\n\t" + key + "='" + str(value) + "'"
        else:
            args_str += ",\n\t" + key + "=" + str(value)
    return args_str


def write_postprocess_PM(fh, args):
    """Write args in fh. """
    write(fh, "postprocess_PM(" + args + ")")


def write_postprocess_GE(fh, args):
    """Write args in fh. """
    write(fh, "postprocess_GE(" + args + ")")


def write_save(fh, varname, prefix, **kwargs):
    write(fh, "from PMMPIESM.setup import save")
    save_args = eval_args_str(varname, **kwargs)
    write(fh, "save("+prefix+","+save_args+", prefix='"+prefix+"')")


def write_compute_skill(fh, **kwargs):
    argstring = eval_args('u',**kwargs)[1:]
    write(fh, "from PMMPIESM.predictability import compute_predictability_horizon")
    write(fh, "skill, threshold, ph = compute_predictability_horizon(ds, control"+argstring+")")


def write_open_da(fh, varname, **kwargs):
    path_args = eval_args_str_name(varname, **kwargs)
    write(fh, "import xarray as xr")
    write(fh, "from PMMPIESM.setup import _get_path")
    write(fh, "ds = xr.open_dataset(_get_path(" + path_args + ", prefix='ds'), chunks={})['"+varname+"']")
    write(fh, "control = xr.open_dataset(_get_path(" + path_args + ", prefix='control'), chunks={})['"+varname+"']")


def create_skill_job_python_file(job_file_str, varname, exp='PM', **kwargs):
    """Short summary.

    Args:
        job_file_str (type): Description of parameter `job_file_str`.
        varname (type): Description of parameter `varname`.
        exp (type): Description of parameter `exp`. Defaults to 'PM'.
        **metric (str): Description of parameter `**kwargs`
        **comparison (str): Description of parameter `**kwargs`
        **bootstrap (int): Description of parameter `**kwargs`
        **sig (int): Description of parameter `**kwargs`

    Returns:
        job_py_str (str): name of jobfile to submit.

    """
    job_directory = ''
    job_py_str = job_file_str + '.py'
    job_py = os.path.join(job_directory, job_py_str)
    if os.path.exists(job_py):
        os.remove(job_py)
    with open(job_py, "w") as fh:
        write(fh, "#!/bin/python")
        # open files
        write_open_da(fh, varname, **kwargs)
        # compute skill
        write_compute_skill(fh, **kwargs)
        # save results
        write_save(fh, varname, prefix='skill', **kwargs)
        write_save(fh, varname, prefix='threshold', **kwargs)
        write_save(fh, varname, prefix='ph', **kwargs)
    return job_py_str


def create_postprocessing_job_python_file(job_file_str, varnames, exp='PM', **kwargs):
    #agg_lead_timeseries=True, compute_skill=True, **kwargs):
    """Write python file for postprocessing.

    Args:
        job_file_str (str): Description of parameter `job_file_str`.
        exp (str): Postprocess Grand Ensemble (GE) or Perfect-model (PM)
        varnames (list): Description of parameter `varnames`.
        **kwargs (type): Description of parameter `**kwargs`.

    Returns:
        job_py_str (str): Description of returned object.

    """
    job_directory = ''
    job_py_str = job_file_str + '.py'
    job_py = os.path.join(job_directory, job_py_str)
    if os.path.exists(job_py):
        os.remove(job_py)
    with open(job_py, "w") as fh:
        write(fh, "#!/bin/python")
        if exp is 'PM':
            setup_PM_postprocess(fh)
            write_postprocess_PM(fh, eval_args(varnames, **kwargs))
        elif exp is 'GE':
            setup_GE_postprocess(fh)
            write_postprocess_GE(fh, eval_args(varnames, **kwargs))
    return job_py_str


def s_lead_timeseries(varnames, job_name=None, exp='PM', test=False, tasks_per_node=1, partition='shared', time='2:00:00', **kwargs):
    """Postprocess PM or GE via slurm.

    Example:
        s_lead_timeseries(['fgco2','spco2'],
            exp='PM', test=False,
            enslist=[3014, 3061], curv=True, model='hamocc',
            outdatatype='data_2d_mm', ending='.nc')

        s_lead_timeseries(['co2flux','pco2'],
            exp='GE', test=False,
            enslist=[1, 2, 3], memberlist=['rcp26', 'rcp45', 'rcp85'], timestr='20[0-1]*', curv=True, model='hamocc',
            outdatatype='data_2d_mm', ending='.nc')

    Creates:
        job_name.py (python script)
        job_name.log (log)
        job_name.job (job_file)

    Args:
        job_name (type): Description of parameter `job_name`.
        varnames (list): List of `varnames`.
        exp (str): Postprocess Grand Ensemble (GE) or Perfect-model (PM).
        test (type): Description of parameter `test`. Defaults to False.
        **kwargs (type): Description of parameter `**kwargs`.

    Returns:
        type: Description of returned object.

    """
    if job_name is None:
        job_name = 's_lead_timeseries_' + exp
        for key, value in kwargs.items():
            job_name += "_" + key + "_" + str(value)
    send_to_slurm(
        create_postprocessing_job_python_file(job_name, varnames, exp=exp, **kwargs),
        job_name,
        test=test, tasks_per_node=tasks_per_node, partition=partition, time=time)


def s_skill(varnames, job_name=None, exp='PM', test=False, tasks_per_node=12, partition='prepost', time='0:20:00', **kwargs):
    """Postprocess PM or GE via slurm.

    Example:
        s_skill(['fgco2','spco2'],
            exp='PM', test=False,
            enslist=[3014, 3061], curv=True, model='hamocc',
            outdatatype='data_2d_mm', ending='.nc')

        s_skill(['co2flux','pco2'],
            exp='GE', test=False,
            enslist=[1, 2, 3], memberlist=['rcp26', 'rcp45', 'rcp85'], timestr='20[0-1]*', curv=True, model='hamocc',
            outdatatype='data_2d_mm', ending='.nc')

    Creates:
        job_name.py (python script)
        job_name.log (log)
        job_name.job (job_file)

    Args:
        job_name (type): Description of parameter `job_name`.
        varnames (type): Description of parameter `varnames`.
        exp (str): Postprocess Grand Ensemble (GE) or Perfect-model (PM).
        test (type): Description of parameter `test`. Defaults to False.
        **kwargs (type): Description of parameter `**kwargs`.

    Returns:
        type: Description of returned object.

    """
    if job_name is None:
        job_name = 's_skill_' + str(varnames) + '_' + exp
        for key, value in kwargs.items():
            job_name += "_" + key + "_" + str(value)
    send_to_slurm(
        create_skill_job_python_file(job_name, varnames, exp=exp, **kwargs),
        job_name,
        test=test, tasks_per_node=tasks_per_node, partition=partition, time=time)
