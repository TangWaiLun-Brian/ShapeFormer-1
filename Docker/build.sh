#sudo docker tag pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel xgdl_base
sudo docker build --build-arg http_proxy=http://proxy.cse.cuhk.edu.hk:8000 --build-arg https_proxy=http://proxy.cse.cuhk.edu.hk:8000 --tag shapeformer:latest . 
