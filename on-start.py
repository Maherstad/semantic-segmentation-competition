##part-1 clone the github repo

import os

os.system('python3 -m pip install gitpython')

import git
from git import RemoteProgress

repo = git.Repo.clone_from("https://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@github.com/Maherstad/semantic-segmentation-and-domain-adaptation.git", "git_repo") 


##part-2 import the raw data

os.system('pip install requests')
os.system('pip install tqdm')

import requests
from tqdm import tqdm

train='https://cos-hdsign-simv-defi.storage-eb4.cegedim.cloud/cos-hdsign-simv-defi/data_flair-one/flair-one_train.zip'
test='https://cos-hdsign-simv-defi.storage-eb4.cegedim.cloud/cos-hdsign-simv-defi/data_flair-one/flair-one_test.zip'
metadata='https://cos-hdsign-simv-defi.storage-eb4.cegedim.cloud/cos-hdsign-simv-defi/data_flair-one/flair-one_metadata.zip'


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        print(save_path)
        for chunk in tqdm(r.iter_content(chunk_size=chunk_size)):
            fd.write(chunk)
            
for i,j in [('train',train),('test',test),('metadata',metadata)]:
    download_url(j,f'/workspace/git_repo/{i}.zip',512)


#create the virtual environment
import subprocess

print('creating virtual ennironvment...')
create_env = subprocess.Popen(["conda", "create", "--name", "venv","python=3.9"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = create_env.communicate(input=b'y\n')

if create_env.returncode == 0:
    print("Virtual environment created successfully, type conda env list to take a look")
else:
    print("Virtual environment creation failed")
    print("Error: ", err.decode())
    
## install zip 
print('installing zip...')

install_zip = subprocess.Popen(["sudo", "apt-get", "install", "zip"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = install_zip.communicate(input=b'y\n')

if install_zip.returncode == 0:
    print("zip installed successfully")
else:
    print("zip installation failed")
    print("Error: ", err.decode())


print('installing dependencies in the virtual environment...')
    
import subprocess
import os
#subprocess.check_output(['ls', '-l'])

def execute_commands(cmds:list):
    for cmd in cmds:
        print(f'ATTEMPING TO EXECUTE : {cmd}')
        try:
            result=subprocess.Popen(cmd,shell=True) #.split()
            #print(result.stdout.decode(), result.stderr.decode())
            result.wait()
            print(f'successful execution of {cmd}')
        except:
            print('command was not executed')
            

os.chdir("/workspace/git_repo/")

cmds_p1=['source activate venv',
         'pip install -r requirements.txt',
         'ipython kernel install --user --name=venv',
         'mkdir raw_dataset',
         'mv test.zip raw_dataset',
         'mv train.zip raw_dataset',
         'unzip metadata.zip',
         'rm metadata.zip'
        ]
execute_commands(cmds_p1)


print('unzipping the raw data...(this might take a while)')


os.chdir('/workspace/git_repo/raw_dataset/')    
cmds_p2=['unzip test.zip','unzip train.zip']
execute_commands(cmds_p2)

os.chdir('/workspace/git_repo/raw_dataset/')    
cmds_p3=['rm test.zip','rm train.zip']
execute_commands(cmds_p3)








