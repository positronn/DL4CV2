# envs.sh

# activate python virtual environment
conda activate vision

curpath=$(pwd)
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONPATH=$curpath