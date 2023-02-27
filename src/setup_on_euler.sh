# make sure miniconda is setup

# name of env to use
export CENV="ut"

# load server modules
env2lmod
module load gcc/6.3.0 python_gpu/3.8.5 eth_proxy cuda/11.3.1

SCRIPTPATH="$( cd -- "$(dirname $BASH_SOURCE)" >/dev/null 2>&1 ; pwd -P )"
PROJECTPATH=$(realpath $SCRIPTPATH"/..")
export PYTHONPATH=$PROJECTPATH

# ensure we have conda
if test ! -f $HOME/miniconda3/bin/activate;
then
	echo "Please install miniconda"
	return
fi

# load conda
source $HOME/miniconda3/bin/activate

# make sure to close all active conda envs to prevent bugs
for i in $(seq ${CONDA_SHLVL}); do
    conda deactivate
done

# if we don't have the env we want, create it
if [[ ! $(conda info --envs | grep ${CENV}) ]];
then
	conda create -n $CENV python=3.8.5
fi

# load env
conda activate $CENV
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/lib/

# install torch
#pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install tensorboard

cd $PROJECTPATH"/Experiments"
mkdir -p $PROJECTPATH"/Experiments/models/UT1/mnist/"
