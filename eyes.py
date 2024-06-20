import cv2

def eyes():
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eyeCascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 640) 
    cap.set(4, 480) 
    
    minW = 0.1 * cap.get(3)
    minH = 0.1 * cap.get(4)

    while 1:
        ret, img = cap.read()
        if ret:
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            font = cv2.FONT_HERSHEY_SIMPLEX

            faces = faceCascade.detectMultiScale(
                frame,
                1.2, 5,
                minSize=(int(minW), int(minH)),
            )
            
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (10, 159, 255), 2)
                frame_tmp = img[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1, :]
                frame = frame[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1]
                eyes = eyeCascade.detectMultiScale(
                    frame,
                    1.2, 5,
                    minSize=(int(minW), int(minH)),
                )
                if len(eyes) == 0:
                    tt = "no eyes!"
                    cv2.putText(img, tt, (0, 0) , font, 1, (255, 255, 255), 2)
                    print('no eyes!')
                else:
                    tt = "eyes!"
                    cv2.putText(img, tt, (0, 0), font, 1, (255, 255, 255), 2)
                    print('eyes!!!')
                frame_tmp = cv2.resize(frame_tmp, (400, 400), interpolation=cv2.INTER_LINEAR)
                cv2.imshow('Face Recognition', frame_tmp)
            waitkey = cv2.waitKey(1)
            if waitkey == ord('q') or waitkey == ord('q'):
                cv2.destroyAllWindows()
                break