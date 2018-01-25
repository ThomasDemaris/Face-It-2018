#!/usr/bin/env python

__author__ = 'Aleksandar Gyorev'
__email__  = 'a.gyorev@jacobs-university.de'

import os
import cv2
import sys
import shutil
import random
import numpy as np
import pickle
import datetime
import sqlite3
from trainDatabase import Eigenfaces


class FindFace(object):   

    def __init__(self):
       #Read eigenfaces database
        with open('database.pkl', 'rb') as input:
            efaces = pickle.load(input)
        self.mean_img_col = efaces.mean_img_col
        self.mn = efaces.mn
        self.evectors = efaces.evectors
        self.W = efaces.W
        self.img_number_per_id = efaces.img_number_per_id
        self.faces_dir = efaces.faces_dir
        self.faces_count = efaces.faces_count
        self.count_closest = 1
            
    """
    Classify an image to one of the eigenfaces.
    """
    def classify(self, path_to_img):
        img = cv2.imread(path_to_img, 0)                                        # read as a grayscale image
        print(path_to_img)
        img_col = np.array(img, dtype='float64').flatten()                      # flatten the image
        img_col -= self.mean_img_col                                            # subract the mean column
        img_col = np.reshape(img_col, (self.mn, 1))                             # from row vector to col vector

        S = self.evectors.transpose() * img_col                                 # projecting the normalized probe onto the
                                                                                # Eigenspace, to find out the weights

        diff = self.W - S                                                       # finding the min ||W_j - S||
        norms = np.linalg.norm(diff, axis=0)
        
        closest_face_id = np.argmin(norms) + self.count_closest                 # the id [0..240) of the minerror face to the sample
        self.count_closest += 1                                      
        print("Closest_id = " +str(closest_face_id))
        

        count = 0
        for i in range(1, self.faces_count + 1):
            count += self.img_number_per_id[i]
            if closest_face_id <= count:
                face_id = i
                break

        
        return face_id                  # return the faceid (1..40)

    """
    Evaluate the model using the 4 test faces left
    from every different face in the AT&T set.
    """
    def evaluate(self):
        print ('> Evaluating the database')
        results_file = os.path.join('results', 'evaluation_results.txt')               # filename for writing the evaluating results in
        f = open(results_file, 'w')                                             # the actual file

        test_count = 1 * self.faces_count                         # number of all AT&T test images/faces
        test_correct = 0
        for face_id in range(1, self.faces_count + 1):
            print("------")
            print("IMG NUMBER = " + str(self.img_number_per_id[face_id]))
            if self.img_number_per_id[face_id] > 1:
                for test_id in range(1, 2):#self.img_number_per_id[face_id]+1):
                    path_to_img = os.path.join(self.faces_dir,
                            's' + str(face_id), str(test_id) + '.pgm')          # relative path

                    result_id = self.classify(path_to_img)
                    result = (int(result_id) == int(face_id))
                    print("Face Id : "+str(face_id) + " --- Found Id : "+str(result_id))

                    if result == True:
                        test_correct += 1
                        f.write('image: %s\nresult: correct\n\n' % path_to_img)
                    else:
                        f.write('image: %s\nresult: wrong, got %2d (face_id = %2d)\n\n' %
                                (path_to_img, result_id, face_id))

        print ('> Evaluating faces from database ended')
        self.accuracy = float(100. * test_correct / test_count)
        print ('Correct: ' + str(self.accuracy) + '%')
        f.write('Correct: %.2f\n' % (self.accuracy))
        f.close()                                                               # closing the file

    """
    Evaluate test faces
    """
    def evaluate_celebrities(self, celebrity_dir='.'):
        print ('> Evaluating test matches started')
        #For each test face
        for img_name in os.listdir(celebrity_dir):                             
            path_to_img = os.path.join(celebrity_dir, img_name)

            img = cv2.imread(path_to_img, 0)                                    # read as a grayscale image
            img_col = np.array(img, dtype='float64').flatten()                  # flatten the image
            img_col -= self.mean_img_col                                        # subract the mean column
            img_col = np.reshape(img_col, (self.mn, 1))                         # from row vector to col vector

            S = self.evectors.transpose() * img_col                             # projecting the normalized probe onto the
                                                                                # Eigenspace, to find out the weights

            diff = self.W - S                                                   # finding the min ||W_j - S||
            norms = np.linalg.norm(diff, axis=0)
            top5_ids = np.argpartition(norms, 5)[:5]                            # first five elements: indices of top 5 matches in AT&T set

            name_noext = os.path.splitext(img_name)[0]                          # the image file name without extension

            #Create corresponding result directory
            now = datetime.datetime.now()
            date = now.strftime("%Y-%m-%d_%Hh%Mm%Ss_")
            result_dir = '.\\results\\'+date+img_name
            os.makedirs(result_dir)                                        
            #Create text file to store results
            result_file = os.path.join(result_dir, 'results.txt')              
            
            f = open(result_file, 'w')                                          # open the results file for writing
            f.write('Test image : ' + img_name +'\n\n')
            
            index = 1
            #For each top id, find the corresponding face id and subimage id
            for top_id in top5_ids:
                count = 0
                for i in range(1, self.faces_count + 1):
                    count += self.img_number_per_id[i]
                    if top_id <= count:
                        face_id = i
                        break
                if index == 1:
                    top_face_id = face_id
                subface_id = top_id - count + self.img_number_per_id[face_id]   # getting the exact subimage from the face

                path_to_found_img = os.path.join(self.faces_dir,
                        's' + str(face_id), str(subface_id) + '.pgm')           # relative path to the top5 face

                shutil.copyfile(path_to_found_img,                              # copy the top face from source
                        os.path.join(result_dir, str(top_id) + '.pgm'))         # to destination

                f.write(str(index) + '.   Face_id: ' + str(face_id)+ ', id: %3d, score: %.6f\n' % (top_id, norms[top_id]))     # write the id and its score to the results file
                index = index + 1
            f.write('\n\nThis is probably face ' + str(top_face_id) +'\n')
            f.close()                                                           # close the results file

            shutil.copyfile(path_to_img,                                        # copy the tested face image
                        os.path.join(result_dir, 'test_face.pgm'))
            print('> Tested image : '+str(img_name))
            print('> Face recognized : '+str(top_face_id)+'\n')

        #TODO
        #Read name database
        #conn = sqlite3.connect('names.db')
        #c = conn.cursor()
        #c.execute('SELECT Name FROM Names WHERE Id=41')
        #user1 = c.fetchone()
        #print(user1)

        print ('> Evaluating matches ended')
        


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print ('Usage: python2.7 eigenfaces.py ' \
            + '<test faces dir>')
        sys.exit(1)

    if not os.path.exists('results'):                                           # create a folder where to store the results
        os.makedirs('results')
    else:
        shutil.rmtree('results')                                                # clear everything in the results folder
        os.makedirs('results')

    findFace = FindFace()
    if len(sys.argv) == 3 :
        findFace.evaluate()                                                     # evaluate our model
    else:
        findFace.evaluate_celebrities(str(sys.argv[1]))

   


