# This was done for ubuntu default on aws.
# Packages/defaults may be slightly different on other distributions.

sudo apt-get update
sudo apt-get upgrade
sudo apt install python3-pip
sudo apt install python3-virtualenv

pip install virtualenv
virtualenv ./.venv/profile-generator
source ./.venv/profile-generator/bin/activate