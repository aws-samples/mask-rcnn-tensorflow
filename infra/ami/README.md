# README
## Upgrading protoc to 3.6.1 for Horovod install

Required on DLAMI 21.2

```
pip uninstall -y protobuf

rm /home/ubuntu/anaconda3/envs/tensorflow_p36_13rc1/bin/protoc
rm -r /home/ubuntu/anaconda3/envs/tensorflow_p36_13rc1/include/google/protobuf
rm /home/ubuntu/anaconda3/envs/tensorflow_p36_13rc1/lib/python3.6/site-packages/protobuf-3.6.0-py3.6-nspkg.pth
rm /home/ubuntu/anaconda3/bin//protoc

wget https://github.com/google/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip
mkdir -p /home/ubuntu/protoc
mv protoc-3.6.1-linux-x86_64.zip /home/ubuntu/protoc/protoc-3.6.1-linux-x86_64.zip
unzip /home/ubuntu/protoc/protoc-3.6.1-linux-x86_64.zip -d protoc
sudo mv /home/ubuntu/protoc/bin/protoc /home/ubuntu/anaconda3/envs/tensorflow_p36_13rc1/bin/protoc
sudo mv /home/ubuntu/protoc/include/* /home/ubuntu/anaconda3/envs/tensorflow_p36_13rc1/include
pip install protobuf==3.6.1