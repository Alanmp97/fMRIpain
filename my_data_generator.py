# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:23:53 2024

@author: damia
"""

import tensorflow as tf
import nibabel as nib
import numpy as np
import os
import cv2
import skimage
import scipy
#import tensorflow as tf
from tensorflow.keras import layers

class FILES_and_LABELS():
    def __init__(self, subjects, sessions, MRI_type, functional_type):
        
        self.sessions = sessions
        self.sub = subjects
        
        ID_subjects = []
        for sub in subjects:
            if 0 < int(sub) < 182:
                subj_ID = 'sub-' + str.zfill(str(sub), 3)
                ID_subjects.append(subj_ID)
        self.subjects = ID_subjects

        ID_sessions = []
        for ses in sessions:
            if 0 < int(ses) <= 3:
                ses_ID = 'ses-' + str.zfill(str(ses), 2)
                ID_sessions.append(ses_ID)
        self.sess = ID_sessions
        
        self.MRI_type = MRI_type
        self.functional_type = functional_type
    

    def get_label(self, sess):
        if sess == 'ses-01':
            label = 0
        elif sess == 'ses-02':
            label = 1
        elif sess == 'ses-03':
            label = 2   
        return label
        
    def get_ID_filenames(self):
        #Funcional_type -> rest o dist
        #FMRI_type -> func o anat
        files = []
        label_files = []
        rootpath = "C:/Users/"+os.getlogin()+"/OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey/rawdata/"
        #label_table = pd.read_table(rootpath + '/participants.tsv', encoding='utf-8')
        for subj in self.subjects:
            for sess in self.sess:
                if self.MRI_type == 'anat':
                    file = subj + '/' + sess + '/anat/' + subj + '_' + sess +  '_T1w.nii'
                elif self.MRI_type == 'func':
                    file = subj + '/' + sess + '/func/' + subj + '_' + sess + '_task-' + self.functional_type + '_bold.nii'
                if os.path.exists(rootpath + file):
                    files.append(file)
                    label = self.get_label(sess)
                    label_files.append(label)
    
        return files, label_files
    
    def get_mask_and_bold(self):
        files = []
        #p = "C:/Users/"+os.getlogin()+"/OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey/"
        if os.getlogin() == "damia":
            for i in self.sessions:
                for j in self.sub:
                    if j <= 68:
                        #image = p+"rabies/preprocess_batch-001_rest/bold_datasink/commonspace_bold/_scan_info_subject_id"+str.zfill(str(j), 3)+".session"+str.zfill(str(i), 2)+"_split_name_sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_desc-o_T2w/_run_None/sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_task-rest_desc-oa_bold_autobox_combined.nii.gz"
                        #mask =  p+"rabies/preprocess_batch-001_rest/bold_datasink/commonspace_mask/_scan_info_subject_id"+str.zfill(str(j), 3)+".session"+str.zfill(str(i), 2)+"_split_name_sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_desc-o_T2w/_run_None/sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_task-rest_desc-oa_bold_autobox_EPI_brain_mask.nii.gz"
                        mask =  "E:/rabies/preprocess_batch-001_rest/bold_datasink/commonspace_mask/_scan_info_subject_id"+str.zfill(str(j), 3)+".session"+str.zfill(str(i), 2)+"_split_name_sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_desc-o_T2w/_run_None/sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_task-rest_desc-oa_bold_autobox_EPI_brain_mask.nii.gz"
                        image = "E:/rabies/preprocess_batch-001_rest/bold_datasink/commonspace_bold/_scan_info_subject_id"+str.zfill(str(j), 3)+".session"+str.zfill(str(i), 2)+"_split_name_sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_desc-o_T2w/_run_None/sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_task-rest_desc-oa_bold_autobox_combined.nii.gz"
                        files.append([image,mask])
                    elif j > 68 and j <= 135:
                        #image = p+"rabies/preprocess_batch-002_rest/bold_datasink/commonspace_bold/_scan_info_subject_id"+str.zfill(str(j), 3)+".session"+str.zfill(str(i), 2)+"_split_name_sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_desc-o_T2w/_run_None/sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_task-rest_desc-oa_bold_autobox_combined.nii.gz"
                        #mask =  p+"rabies/preprocess_batch-002_rest/bold_datasink/commonspace_mask/_scan_info_subject_id"+str.zfill(str(j), 3)+".session"+str.zfill(str(i), 2)+"_split_name_sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_desc-o_T2w/_run_None/sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_task-rest_desc-oa_bold_autobox_EPI_brain_mask.nii.gz"
                        mask =  "E:/rabies/preprocess_batch-002_rest/bold_datasink/commonspace_mask/_scan_info_subject_id"+str.zfill(str(j), 3)+".session"+str.zfill(str(i), 2)+"_split_name_sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_desc-o_T2w/_run_None/sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_task-rest_desc-oa_bold_autobox_EPI_brain_mask.nii.gz"
                        image = "E:/rabies/preprocess_batch-002_rest/bold_datasink/commonspace_bold/_scan_info_subject_id"+str.zfill(str(j), 3)+".session"+str.zfill(str(i), 2)+"_split_name_sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_desc-o_T2w/_run_None/sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_task-rest_desc-oa_bold_autobox_combined.nii.gz"
                        files.append([image,mask])
        elif os.getlogin() == "gdaalumno":
            for i in self.sessions:
                for j in self.sub:
                    if j <= 68:
                        #image = p+"rabies/preprocess_batch-001_rest/bold_datasink/commonspace_bold/_scan_info_subject_id"+str.zfill(str(j), 3)+".session"+str.zfill(str(i), 2)+"_split_name_sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_desc-o_T2w/_run_None/sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_task-rest_desc-oa_bold_autobox_combined.nii.gz"
                        #mask =  p+"rabies/preprocess_batch-001_rest/bold_datasink/commonspace_mask/_scan_info_subject_id"+str.zfill(str(j), 3)+".session"+str.zfill(str(i), 2)+"_split_name_sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_desc-o_T2w/_run_None/sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_task-rest_desc-oa_bold_autobox_EPI_brain_mask.nii.gz"
                        mask =  "C:/Users/gdaalumno/Desktop/rabies/preprocess_batch-001_rest/bold_datasink/commonspace_mask/_scan_info_subject_id"+str.zfill(str(j), 3)+".session"+str.zfill(str(i), 2)+"_split_name_sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_desc-o_T2w/_run_None/sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_task-rest_desc-oa_bold_autobox_EPI_brain_mask.nii.gz"
                        image = "C:/Users/gdaalumno/Desktop/rabies/preprocess_batch-001_rest/bold_datasink/commonspace_bold/_scan_info_subject_id"+str.zfill(str(j), 3)+".session"+str.zfill(str(i), 2)+"_split_name_sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_desc-o_T2w/_run_None/sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_task-rest_desc-oa_bold_autobox_combined.nii.gz"
                        files.append([image,mask])
                    elif j > 68 and j <= 135:
                        if j == 124 and i == 3:
                            continue
                        else:
                            #image = p+"rabies/preprocess_batch-002_rest/bold_datasink/commonspace_bold/_scan_info_subject_id"+str.zfill(str(j), 3)+".session"+str.zfill(str(i), 2)+"_split_name_sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_desc-o_T2w/_run_None/sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_task-rest_desc-oa_bold_autobox_combined.nii.gz"
                            #mask =  p+"rabies/preprocess_batch-002_rest/bold_datasink/commonspace_mask/_scan_info_subject_id"+str.zfill(str(j), 3)+".session"+str.zfill(str(i), 2)+"_split_name_sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_desc-o_T2w/_run_None/sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_task-rest_desc-oa_bold_autobox_EPI_brain_mask.nii.gz"
                            mask =  "C:/Users/gdaalumno/Desktop/rabies/preprocess_batch-002_rest/bold_datasink/commonspace_mask/_scan_info_subject_id"+str.zfill(str(j), 3)+".session"+str.zfill(str(i), 2)+"_split_name_sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_desc-o_T2w/_run_None/sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_task-rest_desc-oa_bold_autobox_EPI_brain_mask.nii.gz"
                            image = "C:/Users/gdaalumno/Desktop/rabies/preprocess_batch-002_rest/bold_datasink/commonspace_bold/_scan_info_subject_id"+str.zfill(str(j), 3)+".session"+str.zfill(str(i), 2)+"_split_name_sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_desc-o_T2w/_run_None/sub-"+str.zfill(str(j), 3)+"_ses-"+str.zfill(str(i), 2)+"_task-rest_desc-oa_bold_autobox_combined.nii.gz"
                            files.append([image,mask])
                    
        return files
    
class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(self, df,
                 batch_size,
                 input_size=(100, 128, 128, 22),
                 shuffle=True,
                 format = "rgb",
                 classes = None,
                 num_class = None,
                 vols = 600,
                 rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                 zoom_range=0.1,  # Randomly zoom image
                 width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                 height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                 horizontal_flip=True,  # randomly flip images
                 vertical_flip=False,
                 augmentation = False):
        #df es una lista con los paths de las sesiones despues del "rawdata/"
        
        self.df = df.copy()
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        if os.getlogin() == "damia":
            self.path = "E:/rawdata/"
        else:
            self.path =  "C:/Users/"+os.getlogin()+"/OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey/rawdata/"
        self.n = len(self.df)
        self.format = format
        self.vols = vols
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.num_class = num_class
        self.classes = classes
        self.augment = augmentation
        
        
    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.df),dtype=int)
            np.random.shuffle(indices)
            self.df = np.array(self.df)[indices]
    
    def __get_input(self, file):
        if self.format != "just_brain" and self.format != "just_brain_M2D":
            #temp = nib.load("E:/rawdata/"+file)
            temp = nib.load(self.path+file)
            image_arr = []
            #image_arr = temp.dataobj[:,:,:,:self.vols]
            for k in range(0,int(620/self.vols)*self.vols,int(620/self.vols)):
                image_arr.append(temp.dataobj[:,:,:,k])

        if self.format == "rgb":
            image_arr = tf.transpose(image_arr, (0,3,1,2))
            image_arr = np.reshape(image_arr, (-1,128,128,1))
            #Z-scoring
            image_arr = (image_arr - np.mean(image_arr))/np.std(image_arr)
            X_rgb = []
            for i in range(len(image_arr)):
                X_rgb.append(cv2.cvtColor(image_arr[i].astype(np.uint8),cv2.COLOR_GRAY2RGB))

            X_rgb = np.array(X_rgb).reshape([-1, 128, 128, 3, 1])
            return np.array(X_rgb)

        elif self.format == "vol":
            #image_arr = tf.transpose(image_arr, (3,0,1,2))
            image_arr = (image_arr - np.mean(image_arr))/np.std(image_arr)
            return np.array(image_arr)
            

        elif self.format == "grayscale":
            #image_arr = tf.transpose(image_arr, (3,2,0,1))
            image_arr = tf.transpose(image_arr, (0,3,1,2))
            image_arr = np.reshape(image_arr, (-1,128,128,1))
            return np.array(image_arr)
        
        elif self.format == "just_brain":
            img = nib.load(file[0])
            mask = nib.load(file[1])
            
            data = img.get_fdata()
            maskdata = mask.get_fdata()
            
            data = np.transpose(data, (3,0,1,2))
            a = [i for i in range(0,int(620/self.vols)*self.vols,int(620/self.vols))]
            data = data[a]
            
            image_arr = maskdata*data
            
            #Z-scoring
            image_arr = (image_arr - np.mean(image_arr))/np.std(image_arr)
            #histogram equalization
            hist = scipy.ndimage.histogram(image_arr, min = 0,
                             max = 255,
                             bins =256)
            cdf = hist.cumsum()/hist.sum()
            image_arr = cdf[np.array(image_arr,dtype="uint8")] * 255
            
            return np.array(image_arr)
        
        elif self.format == "M2D":
            #Z-scoring
            image_arr = (image_arr - np.mean(image_arr))/np.std(image_arr)
            #histogram equalization
            hist = scipy.ndimage.histogram(image_arr, min = 0,
                             max = 255,
                             bins =256)
            cdf = hist.cumsum()/hist.sum()
            image_arr = cdf[np.array(image_arr,dtype="uint8")] * 255
            #image_arr_left = np.transpose(image_arr, (0, 2, 3, 1))
            ##image_arr_front = np.transpose(image_arr, (0, 1, 3, 2))
            #t = []
            ##t.append(image_arr)
            #t.append(image_arr_left)
            #t.append(image_arr_front)
            
            return np.array(image_arr)
        
        elif self.format == "M2D_VGG16":
            #Z-scoring
            image_arr = (image_arr - np.mean(image_arr))/np.std(image_arr)
            image_arr = skimage.transform.resize(image_arr, (self.vols,128, 128, 32))
            
            #image_arr_left = np.transpose(image_arr, (0, 2, 3, 1))
            ##image_arr_front = np.transpose(image_arr, (0, 1, 3, 2))
            #t = []
            ##t.append(image_arr)
            #t.append(image_arr_left)
            #t.append(image_arr_front)
            
            return np.array(image_arr)
        
        elif self.format == "just_brain_M2D":
            img = nib.load(file[0])
            mask = nib.load(file[1])
            
            data = img.get_fdata()
            maskdata = mask.get_fdata()
            
            data = np.transpose(data, (3,0,1,2))
            a = [i for i in range(0,int(620/self.vols)*self.vols,int(620/self.vols))]
            data = data[a]
            
            image_arr = maskdata*data
            
            #Z-scoring
            image_arr = (image_arr - np.mean(image_arr))/np.std(image_arr)
            #histogram equalization
            hist = scipy.ndimage.histogram(image_arr, min = 0,
                             max = 255,
                             bins =256)
            cdf = hist.cumsum()/hist.sum()
            image_arr = cdf[np.array(image_arr,dtype="uint8")] * 255
            
            return np.array(image_arr)
            
            
            
            
    def __get_output(self, file):
        male = ['sub-057',
                 'sub-059',
                 'sub-060',
                 'sub-073',
                 'sub-074',
                 'sub-093',
                 'sub-094',
                 'sub-095',
                 'sub-096',
                 #'sub-097',
                 'sub-098',
                 'sub-099',
                 'sub-100']
        female = ['sub-049',
                 'sub-050',
                 'sub-051',
                 'sub-052',
                 'sub-065',
                 'sub-066',
                 'sub-077',
                 'sub-078',
                 'sub-079',
                 'sub-080',
                 'sub-081',
                 'sub-082',
                 'sub-083',
                 #'sub-084',
                 ]
        CPHfemale = ['sub-049','sub-050','sub-051','sub-052','sub-065','sub-066','sub-077','sub-078','sub-079','sub-080','sub-081',
               'sub-082','sub-083']
        NAIVEfemale = ['sub-019','sub-020','sub-067','sub-068','sub-124','sub-045','sub-046','sub-047','sub-048','sub-061',
                 'sub-062','sub-063','sub-064','sub-084']
        
        CPHmale = ['sub-057','sub-059','sub-060','sub-073','sub-074','sub-093','sub-094','sub-095','sub-096','sub-098','sub-099',
                   'sub-100']
        NAIVEmale = ['sub-024','sub-028','sub-075','sub-076']
        
        label = []
        if self.classes == "CPHvsNAIVEfemale":
            if self.format == "just_brain":
                for i in CPHfemale:
                    if i in file[0]:
                        label.append(0)
                for j in NAIVEfemale:
                    if j in file[0]:
                        label.append(1)
                    
            elif self.format == "just_brain_M2D":
                for i in CPHfemale:
                    if i in file[0]:
                        label.append(0)
                for j in NAIVEfemale:
                    if j in file[0]:
                        label.append(1)
            else:    
                for i in CPHfemale:
                    if i in file:
                        label.append(0)
                for j in NAIVEfemale:
                    if j in file:
                        label.append(1)
                else:
                    label.append("algo salio mal")
                    
        """Para el siguiente caso, las etiquetas estan invertidas (CPH = 1, Naive = 0)"""
        if self.classes == "CPHvsNAIVEmale":
            if self.format == "just_brain":
                for j in NAIVEmale:
                    if j in file[0]:
                        label.append(0)
                for i in CPHmale:
                    if i in file[0]:
                        if 'ses-02' in file[0] or 'ses-03' in file[0]:
                            label.append(1)
                        else:
                            label.append(0)
                if len(label) == 0:
                    print("El siguiente sujeto genera conflicto con las etiquetas\n",file[0])
                
            elif self.format == "just_brain_M2D":
                for j in NAIVEmale:
                    if j in file[0]:
                        label.append(0)
                for i in CPHmale:
                    if i in file[0]:
                        if 'ses-02' in file[0] or 'ses-03' in file[0]:
                            label.append(1)
                        else:
                            label.append(0)
                if len(label) == 0:
                    print("El siguiente sujeto genera conflicto con las etiquetas\n",file[0])
                
            else:    
                for j in NAIVEmale:
                    if j in file:
                        label.append(0)
                for i in CPHmale:
                    if i in file:
                        if 'ses-02' in file or 'ses-03' in file:
                            label.append(1)
                        else:
                            label.append(0)
                if len(label) == 0:
                    print("El siguiente sujeto genera conflicto con las etiquetas\n",file)
                
                    
        if self.classes == "sessions":
            if self.format == "just_brain":
                if 'ses-01' in file[0]:
                    label.append(0)
                elif 'ses-02' in file[0]:
                    label.append(1)
                elif 'ses-03' in file[0]:
                    label.append(2)
                    
            elif self.format == "just_brain_M2D":
                if 'ses-01' in file[0]:
                    label.append(0)
                elif 'ses-02' in file[0]:
                    label.append(1)
                elif 'ses-03' in file[0]:
                    label.append(2)
            else:    
                if 'ses-01' in file:
                    label.append(0)
                elif 'ses-02' in file:
                    label.append(1)
                elif 'ses-03' in file:
                    label.append(2)
                else:
                    label.append("algo salio mal")
                
        if self.classes == "sex":
            if self.format == "just_brain":
                for i in male:
                    if i in file[0]:
                        label.append(0)
                for j in female:
                    if j in file[0]:
                        label.append(1)
                        
            elif self.format == "just_brain_M2D":
                for i in male:
                    if i in file[0]:
                        label.append(0)
                for j in female:
                    if j in file[0]:
                        label.append(1)
                 
            else:
                for i in male:
                    if i in file:
                        label.append(0)
                for j in female:
                    if j in file:
                        label.append(1)
            
        if self.format == "vol":
            label = label*self.vols

        
        elif self.format == "rgb":
            label = label*self.vols*22
        
        elif self.format == "grayscale":
            label = label*self.vols*22
        
        elif self.format == "just_brain":
            label = label*self.vols
        
        elif self.format == "M2D":
            label = label*self.vols
        
        elif self.format == "M2D_VGG16":
            label = label*self.vols
        
        elif self.format == "just_brain_M2D":
            label = label*self.vols
        return tf.keras.utils.to_categorical(label, num_classes=self.num_class)
        #return label
        

    def __get_data(self, batches):
        # Generates data containing batch_size samples
        path_batch = batches

        y_batch = np.asarray([self.__get_output(y) for y in path_batch])
        y_batch = np.reshape(y_batch, (-1,self.num_class))
        #y_batch = np.reshape(y_batch, (-1))
        
        """eliminar siguientes 2 linea"""
        file = np.asarray([[f]*self.vols for f in path_batch])
        file = np.reshape(file, (-1))
        
        
        indices = np.arange(len(y_batch),dtype=int)
        np.random.shuffle(indices)
        
        y_batch = np.array(y_batch)[indices]
        """eliminar siguiente linea"""
        file = np.array(file)[indices]
        
        
        if self.format == "rgb": 
            X_batch = np.asarray([self.__get_input(x) for x in path_batch])
            X_batch = np.reshape(X_batch, (-1,128,128,3))
            
            X_batch = np.array(X_batch)[indices]
            
            return X_batch, y_batch
            
        elif self.format == "vol":
            X_batch = np.asarray([self.__get_input(x) for x in path_batch])
            X_batch = np.reshape(X_batch, (-1,128,128,22))
            
            X_batch = np.array(X_batch)[indices]
    
            return X_batch, y_batch
            
        elif self.format == "grayscale":
            X_batch = np.asarray([self.__get_input(x) for x in path_batch])
            X_batch = np.reshape(X_batch, (-1,128,128,1))
            
            X_batch = np.array(X_batch)[indices]
            
            return X_batch, y_batch
            
        elif self.format == "just_brain":
            X_batch = np.asarray([self.__get_input(x) for x in path_batch])
            X_batch = np.reshape(X_batch, (-1,48, 81, 48))
            
            X_batch = np.array(X_batch)[indices]
            
            return X_batch, y_batch
            
        elif self.format == "M2D":
            X_batch = np.asarray([self.__get_input(x) for x in path_batch])
            X_batch = np.reshape(X_batch, (-1,128,128,22))
            
            X_batch = np.array(X_batch)[indices]
            
            image_arr_left = np.transpose(X_batch, (0, 2, 3, 1))
            image_arr_front = np.transpose(X_batch, (0, 1, 3, 2))
            
            return [X_batch,image_arr_front,image_arr_left], y_batch
        
        elif self.format == "M2D_VGG16":
            X_batch = np.asarray([self.__get_input(x) for x in path_batch])
            X_batch = np.reshape(X_batch, (-1,128,128,32))
            
            X_batch = np.array(X_batch)[indices]
            
            image_arr_left = np.transpose(X_batch, (0, 2, 3, 1))
            image_arr_front = np.transpose(X_batch, (0, 1, 3, 2))
            
            return [X_batch,image_arr_front,image_arr_left], y_batch
        
        elif self.format == "just_brain_M2D":
            X_batch = np.asarray([self.__get_input(x) for x in path_batch])
            X_batch = np.reshape(X_batch, (-1,48, 81, 48))
            
            X_batch = np.array(X_batch)[indices]
            
            image_arr_left = np.transpose(X_batch, (0, 2, 3, 1))
            image_arr_front = np.transpose(X_batch, (0, 1, 3, 2))
            
            
            return [X_batch,image_arr_front,image_arr_left], y_batch
    
        
        
        #return np.array(X_batch, dtype='uint8'), y_batch
        
    
    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        
        X, y= self.__get_data(batches)   
        
        #indices = np.arange(len(y),dtype=int)
        #np.random.shuffle(indices)
        if self.augment == True:
            data_augmentation = tf.keras.Sequential([
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.2),
                ])
            X = data_augmentation(X)
        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size