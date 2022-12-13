from dataclasses import field
from tkinter import Image
import cv2
import mysql.connector
import os
from PIL import Image,ImageTk
import numpy as np
import time
import csv

stdId = []
stdName =[]
class Attendance:
    def addData():
        try:
            conn=mysql.connector.connect(host="localhost",username="root",password="root",database="attendance")
            mycursor=conn.cursor()
            dep = input("Enter Department: ")
            course = input("Enter Course: ")
            year = input("Enter Year: ")
            semester = input("Enter Semester: ")
            name = input("Enter Name: ")
            id = int(input("Enter Id: "))

            mycursor.execute("INSERT into student values(%s,%s,%s,%s,%s,%s,%s)",(dep,course,year,semester,id,name,"none"))
            conn.commit()
            conn.close()
            print("Details Added!")
        except Exception as e:
            print("Error = "+e)

    def showData():
        conn=mysql.connector.connect(host="localhost",username="root",password="root",database="attendance")
        mycursor=conn.cursor()   
        mycursor.execute("Select * from student")
        data=mycursor.fetchall()

        if len(data)!=0:
            for i in data:
                print(i,end="\n")
            conn.commit()
        conn.close() 

    def deleteData():
        conn=mysql.connector.connect(host="localhost",username="root",password="root",database="attendance")
        mycursor=conn.cursor()  
        mycursor.execute("DELETE from student")
        conn.commit()
        conn.close()

        print("Deleted All Data")

    def generateDataset(uid):
        faceClassifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        def faceCropped(img):
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces=faceClassifier.detectMultiScale(gray,1.3,5)
            #scaling Factor = 1.3
            #minimum Neighbour = 5
            for(x,y,w,h) in faces:
                faceCropped=img[y:y+h,x:x+w]
                return faceCropped

        print("Collection Photo Sample:")
        print("Please Look Into The Camera")
        # time.sleep(1)
        cap = cv2.VideoCapture(0)
        imgId=0
        while True:
            ret,myframe = cap.read()
            if faceCropped(myframe) is not None:
                imgId+=1
                # time.sleep(1)
                face = cv2.resize(faceCropped(myframe),(450,450))
                face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                filePath = "data/"+str(uid)+"."+str(imgId)+".jpg"
                cv2.imwrite(filePath,face)
                cv2.putText(face,str(imgId),(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
                cv2.imshow("Cropped Face",face)
            if cv2.waitKey(1) == 13 or int(imgId) == 20:
                break
        cap.release()
        cv2.destroyAllWindows()
        print("Data Collected!")


    def trainModel():
        data_dir = ("data")
        path = [os.path.join(data_dir,file) for file in os.listdir(data_dir)]

        faces = []
        ids = []

        for image in path:
            img=Image.open(image).convert('L')
            ImageNp = np.array(img,'uint8')
            x = os.path.split(image)[1]
            id = int(x.split('.')[0])
            
            faces.append(ImageNp)
            ids.append(id)
            cv2.imshow("Training Model",ImageNp)
            cv2.waitKey(1) == 13

        ids=np.array(ids)

        # training the classifier and save
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces,ids)
        clf.write("recognizer.xml")
        cv2.destroyAllWindows()
        print("Model Trained!")


    def registerAtt(id,data):
        if id in stdId:
            pass
        else:
            stdId.append(id)
            stdName.append(data)

    def getAttendanceData():
        print(f"Total Students = {len(stdId)}")
        print(f"Student Ids = {stdId}")
        print(f"Student Names = {stdName}")
        


    # Face Recognition
    def takeAttendance():
        attendance=[]
        def drawBoundary(img,classifier,scaleFactor,minNeighbour,color,text,clf):
            grayImage = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            features = classifier.detectMultiScale(grayImage,scaleFactor,minNeighbour)

            coord=[]

            for (x,y,w,h) in features:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
                id,predict = clf.predict(grayImage[y:y+h,x:x+w])
                # Prediction Formula
                confidence = int((100*(1-predict/300)))

                # Data Fetching
                conn=mysql.connector.connect(host="localhost",username="root",password="root",database="attendance")
                mycursor=conn.cursor()
                mycursor.execute("Select name from student where id="+str(id))
                data = mycursor.fetchone()

                if confidence > 70:
                    filteredData = data[0]
                    Attendance.registerAtt(id,filteredData)


                    cv2.putText(img,f"Name:{data}",(x,y-50),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)
                else:
                    cv2.putText(img,f"Unknown Person",(x,y-50),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,255),2)

                coord = [x,y,w,h]

            return coord

        def recognition(img,clf,faceCascade):
            coord=drawBoundary(img,faceCascade,1.1,10,(255,25,255),"face",clf)
            return img

        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        clf=cv2.face.LBPHFaceRecognizer_create()
        clf.read("recognizer.xml")

        videoCap = cv2.VideoCapture(0)

        while True:
            ret,img = videoCap.read()
            img=recognition(img,clf,faceCascade)
            cv2.imshow("Attendace Tracking",img)

            if cv2.waitKey(1) == 13:
                break
        videoCap.release()
        cv2.destroyAllWindows()

        def registerAttendace(uid):
            prevId = 0
            


if __name__ == '__main__':
    monitor = Attendance
    print("Welcome To Attendance Management System:")
    print("1. Register Student")
    print("2. Take Photo")
    print("3. Train Model")
    print("4. Take Attendance")
    print("5. Show Students")
    print("6. Delete Data")
    print("7. Exit")


    while(1):
        choice = int(input("\nEnter your choice: "))
        if(choice == 1):
            monitor.addData()     
        elif(choice == 2):
            uid = int(input("Enter Student Id: "))
            monitor.generateDataset(uid)
        elif(choice == 3):
            monitor.trainModel()       
        elif(choice == 4):
            monitor.takeAttendance() 
            print("\nToday's Attendance\n")
            monitor.getAttendanceData()
        elif(choice == 5):
            monitor.showData()
        elif(choice == 6):
            monitor.deleteData()
        elif(choice == 7):
            exit()