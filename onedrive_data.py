import os
import fnmatch


class FilesNamesAndTypes:

    def __init__(self, mri_type=2, num_subject=None):
        if 0 < mri_type <= 3:
            self.mri_type = mri_type
        else:
            raise ValueError(f"{mri_type} must be 1 or 2 o 3")

        self.rootPath = "C:/Users/damia/OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey" \
                        "/rawdata"
        
       # self.rootPath = "C:/Users/Dental_User/OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey" \
        #                "/rawdata"
        self.num_subject = num_subject
        self.matches = []

        if self.mri_type == 1:
            self.pattern = "*T1w.nii"
        elif self.mri_type == 2:
            self.pattern = "*rest_bold.nii"
        elif self.mri_type == 3:
            self.pattern = "*dist_bold.nii"

    def select_files_name_type(self):
        if self.num_subject is not None:
            subj = os.listdir(self.rootPath)
            subj = fnmatch.filter(subj, "sub*")
            
            #si usamos una lista para dar los sujetos, [i,j] indicara que se piden los sujetos del i al j
            if type(self.num_subject) == list:
                if 0 <= self.num_subject[-1] <= len(subj):
                    for i in range(self.num_subject[1]-self.num_subject[0]+1):   
                        for root, dirs, files in os.walk(self.rootPath+ "/"+ subj[self.num_subject[0]-1+i]):
                            for filename in fnmatch.filter(files, self.pattern):
                                self.matches.append(os.path.join(root, filename))
                else:
                    raise ValueError(f" num-subject is {self.num_subject} and must be lower than {len(self.matches)} "
                                     f"and greater than 0 ")
            
            #la tupla (5,7,12) indica que se piden los sujetos 5, 7 y 12
            elif type(self.num_subject) == tuple:
                sub = []
                for i in range(len(self.num_subject)):
                    if len(str(self.num_subject[i])) == 1:
                        sub.extend(fnmatch.filter(subj, "sub-00" + str(self.num_subject[i])))
                    elif len(str(self.num_subject[i])) == 2:
                        sub.extend(fnmatch.filter(subj, "sub-0" + str(self.num_subject[i])))
                    else:
                        sub.extend(fnmatch.filter(subj, "sub-" + str(self.num_subject[i])))
                for j in range(len(sub)):
                    for root, dirs, files in os.walk(self.rootPath+ "/"+ sub[j]):
                        for filename in fnmatch.filter(files, self.pattern):
                            self.matches.append(os.path.join(root, filename))
                    
            #si solo se da un valor, se solicita ese sujeto
            else:
                sub = []
                if 0 <= self.num_subject <= len(subj):
                    if len(str(self.num_subject)) == 1:
                        sub.extend(fnmatch.filter(subj, "sub-00" + str(self.num_subject)))
                    elif len(str(self.num_subject)) == 2:
                        sub.extend(fnmatch.filter(subj, "sub-0" + str(self.num_subject)))
                    else:
                        sub.extend(fnmatch.filter(subj, "sub-" + str(self.num_subject)))
                    for root, dirs, files in os.walk(self.rootPath+ "/"+ sub[0]):
                        for filename in fnmatch.filter(files, self.pattern):
                            self.matches.append(os.path.join(root, filename))
                else:
                    raise ValueError(f" num-subject is {self.num_subject} and must be lower than {len(self.matches)} "
                                     f"and greater than 0 ")
        #si num_subject is none, se entran todos los sujetos de la carpeta
        else:
            for root, dirs, files in os.walk(self.rootPath):
                for filename in fnmatch.filter(files, self.pattern):
                    #print(os.path.join(root, filename))
                    self.matches.append(os.path.join(root, filename))
              
    def get_files(self):
        return self.matches
    
    def print_numDataAndLength(self):
        print(f"you select {len(self.matches)} files :")
        print(self.matches, sep='\n')



