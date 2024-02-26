ENV_NAME="bench"
	
conda create -n $ENV_NAME python=3.8 ipykernel nb_conda_kernels
source activate $ENV_NAME

pip install numpy	
pip install pandas
pip install tensorflow
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install scipy
pip install gdown
