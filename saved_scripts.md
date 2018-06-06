# Prep the cloud session
sudo apt-get install screen
sudo pip install pandas
sudo pip install scipy
sudo pip install sklearn


# To train:
`screen -S proj0 ./scripts/proj0.sh`

# To sample:
`python evaluate_rnn_classifications.py --init_dir=model1 --temperature=0.5`

