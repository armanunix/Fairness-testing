FROM ubuntu:20.04

# Dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update
WORKDIR /root
ADD Requirements.txt /root/
ADD clusters /root/clusters
ADD datasets /root/datasets
ADD DICE_data /root/DICE_data
ADD DICE_model /root/DICE_model
ADD DICE_text /root/DICE_text
ADD DICE_tutorial /root/DICE_tutorial
ADD DICE_utils /root/DICE_utils
ADD models /root/models
RUN apt install -y python3 python3-pip
RUN pip install -r Requirements.txt
ADD Raw_data /usr/local/lib/python3.8/dist-packages/aif360/data/
WORKDIR /root/DICE_tutorial/
#RUN python3 DICE_Search_final.py -dataset=census -sensitive_index=[9,8,1] -timeout=300
#RUN python3 DICE_Debugging_2.py -dataset=census -sensitive_index=[9,8,1]
