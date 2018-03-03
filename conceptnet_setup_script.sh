
sudo yum groupinstall "Development tools"
sudo yum install zlib-devel bzip2-devel openssl-devel ncurses-devel sqlite-devel readline-devel tk-devel gdbm-devel db4-devel libpcap-devel xz-devel
sudo yum install python-devel python-nose python-setuptools gcc gcc-gfortran gcc-c++ blas-devel lapack-devel atlas-devel


wget http://python.org/ftp/python/2.7.6/Python-2.7.6.tar.xz
tar xf Python-2.7.6.tar.xz
cd Python-2.7.6
./configure --prefix=/usr/local --enable-unicode=ucs4 --enable-shared LDFLAGS="-Wl,-rpath /usr/local/lib"
sudo make && make altinstall

export VIRTUALENV_PYTHON=/usr/local/bin/python2.7
virtualenv ENV1

source ENV1/bin/activate
pip install numpy
pip install scipy
#---------------------
pip install flask
cd conceptnet5
python setup.py develop
#GUROBI Installation---------------------
cd gurobi650/linux64
python setup.py install
# Change in ~/.bashrc
   export GUROBI_HOME="/opt/gurobi562/linux64"
   export PATH="${PATH}:${GUROBI_HOME}/bin"
   export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
source ~/.bashrc

pip install scikit-learn
pip install gensim

cd data
curl -O  http://conceptnet5.media.mit.edu/downloads/v5.3/conceptnet5_db_5.3.tar.bz2
cd ..
tar jxvf data/conceptnet5_db_5.3.tar.bz2
touch data/db/*
ln -s /grid/11/somak/Image_Riddle/conceptnet5/data ~/.conceptnet5
