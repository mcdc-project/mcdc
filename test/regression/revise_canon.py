import os

for item in os.listdir():
    if os.path.isdir(item):
        print(item)
        os.chdir(item)
        os.system("mv output.h5 answer.h5")
        os.chdir(r"..")
