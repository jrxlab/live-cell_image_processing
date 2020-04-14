import os


def GetFilePath(dir_path):
    
    files= os.listdir(dir_path)
    
    print(files)
    
    for file in files:
        if dir_path[-1] != "/" :
            dir_path=dir_path+"/"
        file_path= dir_path+str(file)
        print(file_path)
       
            
    return file_path


### main
sequence_path = "C:/Users/ch_95/Desktop/data"

file=GetFilePath(sequence_path)