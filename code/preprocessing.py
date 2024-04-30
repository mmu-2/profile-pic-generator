import PIL
import glob
from torchvision import transforms
import os


def onlyresize():
     filepath = f"./{inputfolder}_resized/"
     os.makedirs(filepath, exist_ok=True)


     for filename in glob.glob(f'./{inputfolder}/*.jpg') + glob.glob(f'./{inputfolder}/*.jpeg'):

          img = PIL.Image.open(filename)

          mindim = min(img.size[-2:]) #not sure if channel is included and too lazy to check

          # crop = transforms.RandomCrop(mindim)
          crop = transforms.CenterCrop(mindim)
          resize = transforms.Resize((512, 512))

          fileonly = filename.split('/')[-1]
          fileonly = ".".join(fileonly.split('.')[:-1])
          print(filepath + fileonly + '_resized')
          
          resized = resize(crop(img))
          resized.save(filepath + fileonly + '_resized.jpg', 'JPEG')


def onlycrop():
     filepath = f"./{inputfolder}_processed/"
     os.makedirs(filepath, exist_ok=True)


     for filename in glob.glob(f'./{inputfolder}/*.jpg') + glob.glob(f'./{inputfolder}/*.jpeg'):
          # filepath = filename.split('/')[:-1]
          # filepath = "/".join(filepath) + "/cropped/"

          img = PIL.Image.open(filename)
          crop = transforms.RandomCrop(512)
          resize = transforms.Resize((512, 512))

          fileonly = filename.split('/')[-1]
          fileonly = ".".join(fileonly.split('.')[:-1])
          print(filepath + fileonly + '_cropped')

          
          # elif img.size[0] > 1024 and img.size[1] > 1024:
          if img.size[0] * img.size[1] >= 2048 * 2048:
               crops = 16
          elif img.size[0] * img.size[1] >= 1024 * 2048:
               crops = 8
          elif img.size[0] * img.size[1] >= 1024 * 1024:
               crops = 4
          elif img.size[0] * img.size[1] >= 512 * 1024:
               crops = 2
          else:
               crops = 1

          for i in range(crops):
                    cropped = crop(img)
                    cropped.save(filepath + fileonly + f'_cropped_{i}.jpg', 'JPEG')


# inputfolder = "borderlands"
# inputfolder = "naruto"
inputfolder = "ghibli" # https://www.ghibli.jp/info/013344/
onlyresize()
# onlycrop()