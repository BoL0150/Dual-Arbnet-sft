python main.py --n_GPUs=1  \
               --data_train='RefMRI' \ 
               --name_train='FSPDmattrain' \  # HR的mri在数据目录下的目录名字
               --data_test='RefMRI' \
               --name_test='FSPDmatval' \
               --dir_data='/Data'  \    # 数据目录的路径
               --loss="1*L1+0.05*KLoss"  \
               --model="dualref"  \
               --save=""  \
               --batch_size=6 \
               --patch_size=32 \
               --resume=0  \
               --n_color=2 \
               --rgb_range=1 \
               --ref_mat='fastMRIref_mat' \ # ref的mri在数据目录下的目录名字
               --ref_list='multiname.txt' \ 
               --pre_train=None

python main.py --n_GPUs=8 --data_train=RefMRI --name_train=mattest --data_test=RefMRI --name_test=val_mattest --dir_data=./demodata --loss="1*L1+0.05*KLoss" --model=dualref --save="" --batch_size=150 --patch_size=32 --resume=0 --n_color=2 --rgb_range=1 --ref_mat=ref_mat --ref_list=multiname.txt --pre_train=/home/libo/Dual-ArbNet/experiment/model/model_best.pt
python main.py --n_GPUs=1 --data_train=RefMRI --name_train=mattest --data_test=RefMRI --name_test=val_mattest --dir_data=./demodata --loss="1*L1+0.05*KLoss" --model=dualref --save="" --batch_size=12 --patch_size=32 --resume=0 --n_color=2 --rgb_range=1 --ref_mat=ref_mat --ref_list=multiname.txt --pre_train=None