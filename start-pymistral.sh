#!/bin/bash
#
# Copyright 2018 Deutsches Klimarechenzentrum GmbH
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
# start-jupyter
#
# This script is intended to be used on your local workstation running
# Ubuntu, Fedora Linux or macOS (tested). Other Unix flavors may work
# as well. It allows you to start jupyter notebook or lab on one of
# DKRZ's mistral nodes. The script then opens a ssh tunnel to connect
# your local browser to jupyter.
#
# If you indicate an account with the -A switch, the script will run
# jupyter in a job on dedicated resources. Otherwise jupyter uses a
# shared interactive node.
#
# In case of problems contact Mathis Rosenhauer <rosenhauer@dkrz.de>.
#

set -eufo pipefail

# Default settings
#
# You can change the settings here or override them with command line
# options.

# Project account code.
#
# Jupyter is started on the frontend if this is not set.
SJ_ACCTCODE="mh0033"

# LDAP username
#
# Specify your username on the frontend if it is not your local
# username.
# SJ_USERNAME="$(id -un)"
SJ_USERNAME="m300265"
# Jupyter command
#
# You could change this to lab for example (default: "notebook")
SJ_COMMAND="lab"

# Generate debugging output if set to 1
SJ_DEBUG=0

# Session run time in minutes
SJ_RUNTIME=420

# Ntasks for job
SJ_NTASKS=1

# Partition for job
SJ_PARTITION=shared

# Incfile
#
# If indicated, this file will be sourced prior to running jupyter
# notebook. It has to be located on mistral. Set up the environment
# for starting the correct jupyter here.
SJ_INCFILE="pymistral_preload"

# Frontend host
#
# Must be directly accessible from client. The frontend and the node
# where jupyter is running need a shared home file system.
readonly SJ_FRONTEND_HOST="mistral.dkrz.de"


function clean_up () {
    trap - ERR EXIT
    set +e

    echo
    if [[ -n ${SJ_ACCTCODE} ]]; then
        if [[ -n ${jupyter_id:-} ]]; then
            echo "Removing job ${jupyter_id}."
            ssh_frontend "scancel -Q ${jupyter_id}"
        else
            echo "Job ID not available. Make sure the jupyter " \
                 "job is not running!"
            ssh_frontend "squeue -u ${SJ_USERNAME}"
        fi
    else
        if [[ -n ${jupyter_id:-} ]]; then
            echo "Killing jupyter process ${jupyter_id}..."
            ssh_frontend "kill ${jupyter_id}"
        fi
    fi

    ssh_frontend "rm -f ${jupyter_log}"
    ssh_frontend "" "-O exit"
    rmdir "${ssh_socket_dir}"

    exit
}

function usage () {
    cat <<EOF
Usage: $(basename "$0") [OPTION]

Available values for OPTION include:

  -A acctcode    start a job with acctcode
  -c command     invoke jupyter with command
  -d             check for presence of jupyter
  -i file        source file prior to running jupyter
  -n ntasks      request ntasks tasks for job
  -p partition   run job on partition
  -t time        job runtime
  -u username    use username for login

EOF
}

function parse_options () {
    local option
    while getopts 'A:c:di:n:p:t:u:' option; do
        case ${option} in
            A) SJ_ACCTCODE="$OPTARG"
               ;;
            c) SJ_COMMAND="$OPTARG"
               ;;
            d) SJ_DEBUG=1
               ;;
            i) SJ_INCFILE="$OPTARG"
               ;;
            n) SJ_NTASKS="$OPTARG"
               ;;
            p) SJ_PARTITION="$OPTARG"
               ;;
            t) SJ_RUNTIME="$OPTARG"
               ;;
            u) SJ_USERNAME="$OPTARG"
               ;;
            ?) usage
            exit 1
            ;;
        esac
    done
    readonly SJ_ACCTCODE
    readonly SJ_COMMAND
    readonly SJ_DEBUG
    readonly SJ_INCFILE
    readonly SJ_NTASKS
    readonly SJ_PARTITION
    readonly SJ_RUNTIME
    readonly SJ_USERNAME
}

function ssh_frontend () {
    # Run command on frontend with ssh
    local command="$1"
    local extra_options="${2:-}"
    local options
    options="${extra_options} -o ForwardX11=no \
            -o ControlPath=${ssh_socket_dir}/control:%h:%p:%r"
    ssh ${options} "${SJ_USERNAME}@${SJ_FRONTEND_HOST}" "${command}"
}

function source_incfile() {
    # Add sourcing of incfile to commandline if an incfile was
    # specified
    local commandline="$1"
    if [[ -n ${SJ_INCFILE} ]]; then
        local incfile="${SJ_INCFILE}"
        if [[ ${incfile:0:1} != "/" ]]; then
            incfile="\${HOME}/${incfile}"
        fi
        commandline="source ${incfile}; ${commandline}"
    fi
    echo "${commandline}"
}

function which_jupyter() {
    echo "Looking for Jupyter."
    local which
    which="$(source_incfile "which jupyter")"
    ssh_frontend "/bin/bash -lc \"${which}\""
}

function assemble_commandline () {
    local logfile="$1"

    local commandline="jupyter ${SJ_COMMAND} --no-browser 2>> ${logfile}"
    # If we are not running a job, we have to perform our own scheduling
    if [[ -z ${SJ_ACCTCODE} ]]; then
        commandline="nohup ${commandline} > /dev/null & echo \$!"
    fi
    commandline="$(source_incfile "${commandline}")"
    echo "${commandline}"
}

function submit_jupyter_job () {
    local commandline="$1"
    local logfile="$2"

    ssh_frontend "sbatch" <<EOF
#!/bin/bash -l
#SBATCH --job-name=Jupyter
#SBATCH --partition=${SJ_PARTITION}
#SBATCH --ntasks=${SJ_NTASKS}
#SBATCH --time=${SJ_RUNTIME}
#SBATCH --account=${SJ_ACCTCODE}
#SBATCH --output=/dev/null
#SBATCH --parsable
#SBATCH --dkrzepilog=0
cd \${HOME}
echo "NODE:\${SLURM_JOB_NODELIST}" > ${logfile}
${commandline}
EOF
}

function run_jupyter () {
    local logfile="$1"

    local commandline
    commandline="$(assemble_commandline "${jupyter_log}")"
    # Run commandline in job or directly on frontend
    if [[ -n ${SJ_ACCTCODE} ]]; then
        submit_jupyter_job "${commandline}" "${logfile}"
    else
        ssh_frontend "/bin/bash -ls" <<< "${commandline}"
    fi
}

function extract_from_logs () {
    local pattern="$1"
    local haystack="$2"
    ssh_frontend "/bin/bash -s" <<EOF
while [[ -z \${needle} ]]; do
    sleep 1
    if [[ -f ${haystack} ]]; then
        needle="\$(grep -Pom 1 "${pattern}" "${haystack}")"
    fi
    printf "." >&2
done
printf "\n" >&2
echo "\${needle}"
EOF
}

function get_jupyter_node () {
    local logfile="$1"

    if [[ -n ${SJ_ACCTCODE} ]]; then
        printf "Waiting for job to start" >&2
        extract_from_logs "NODE:\K\w+" "${logfile}"
    else
        echo "${SJ_FRONTEND_HOST}"
    fi
}

function open_tunnel () {
    local node="$1"
    local port="$2"

    if [[ -n ${SJ_ACCTCODE} ]]; then
        # Tunnel to notebook in job needs jump host since nodes
        # usually have no direct external access.
        # Unfortunately, doesn' seem to support connection sharing
        # for the jump host
        # -o StrictHostKeyChecking=accept-new \
        ssh -o ForwardX11=no \
            -J "${SJ_USERNAME}@${SJ_FRONTEND_HOST}" \
            -L "${port}:localhost:${port}" \
            -Nf \
            "${SJ_USERNAME}@${node}"
    else
        ssh_frontend "" "-O forward -L${port}:localhost:${port}"
    fi
}

function show_url() {
    local url="$1"

    echo "Open the following URL with your browser"
    echo "${url}"
}

function launch_browser() {
    local url="$1"

    case "$(uname -s)" in
        Darwin)
            open "${url}" || show_url "${url}"
            ;;
        Linux)
            xdg-open "${url}" || show_url "${url}"
            ;;
        *)
            show_url "${url}"
            ;;
    esac
}

function main () {
    parse_options "$@"
    trap clean_up INT QUIT TERM ERR EXIT

    echo "Establishing ssh master connection."
    # Set up control master for connection sharing
    ssh_socket_dir="$(mktemp -d "${HOME}/.ssh/socket.XXXXX")"
    ssh_frontend "" "-MNf"

    # Create unique output file for jupyter notebook
    jupyter_log="$(ssh_frontend "mkdir -p \${HOME}/.jupyter \
                                && mktemp \${HOME}/.jupyter/jupyter.XXXXX")"
    ssh_frontend "chmod 600 ${jupyter_log}"

    # Check for jupyter
    [[ ${SJ_DEBUG} == 1 ]] && which_jupyter

    jupyter_id="$(run_jupyter "${jupyter_log}")"
    local node
    node="$(get_jupyter_node "${jupyter_log}")"

    # Get notebook url and token from output
    printf "Starting jupyter server" >&2
    local url
    url="$(extract_from_logs "^.*\Khttp://localhost:.+" "${jupyter_log}")"
    ssh_frontend "rm -f ${jupyter_log}"

    local port
    port=${url#*t:}
    port=${port%%/*}

    open_tunnel "${node}" "${port}"
    echo "Established tunnel to ${node}:${port}."

    launch_browser "${url}"
    echo "Press Ctrl-C to stop jupyter and shut down tunnel."

    sleep 604800
}

main "$@"
