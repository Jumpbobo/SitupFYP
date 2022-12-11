
import cv2
import mediapipe as mp
import numpy as np
import time
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

capture = cv2.VideoCapture('./sucai/ywqz_yasuo.MP4')

count=0
direction=0
ptime=0

def calculate_angle(a, b, c):
    # first point
    a = np.array(a)

    # mid point
    b = np.array(b)

    # end point
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])

    angle = np.abs(radians * 180 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle



## setup mediapipe instance
with mp_pose.Pose(model_complexity=1,min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1792, 828))


        # recolor image to RGB

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # make detection
        results = pose.process(image)


        # recolor back toBGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # extract landmarks
        # results.pose_landmarks.landmark[26].visibility 获取点的visibility


        try:

            landmarks = results.pose_landmarks.landmark

            # get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            shoulderRight=[landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbowleft = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            elbowright = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wristleft = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            wristright = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            shouldermid=[(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x+landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x)*0.5,
                            (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y + landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y) * 0.5
                        ]
            mouthmid=[(landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x+landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x)*0.5,
                            (landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y + landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y) * 0.5
                        ]
            pshoulderleft=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                        ]

            hipleft = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            wristleft = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            kneeleft = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankleleft = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            pshouldermid = np.multiply(shouldermid, [640, 480]).astype(int)
            pmouthmid = np.multiply(mouthmid, [640, 480]).astype(int)


            pshoulderleft = np.multiply(pshoulderleft, [1792, 828]).astype(int)
            pelbowleft = np.multiply(elbowleft, [1792, 828]).astype(int)
            pwristleft = np.multiply(wristleft, [1792, 828]).astype(int)

            pshoulderright = np.multiply(shoulderRight, [1792, 828]).astype(int)
            pelbowright = np.multiply(elbowright, [1792, 828]).astype(int)
            pwristright = np.multiply(wristright, [1792, 828]).astype(int)

            phipleft = np.multiply(hipleft, [1792, 828]).astype(int)
            pkneeleft = np.multiply(kneeleft, [1792, 828]).astype(int)
            pankleleft = np.multiply(ankleleft, [1792, 828]).astype(int)


#            cv2.line(image,pshouldermid,pmouthmid,0,2)
#            cv2.line(image,pnose,pmouthmid,0,2)




            # calculate angle
            angle_kua = calculate_angle(pshoulderleft, phipleft, pkneeleft)
            angle_xigai = calculate_angle(phipleft, pkneeleft, pankleleft)
            angle2=calculate_angle(pshoulderleft,pshouldermid,pmouthmid)
            print(angle_kua)



            #print(shouldermid)

            # visualize
            #cv2.putText(image, str(int(angle_kua)),
             #           tuple(np.multiply(elbowright, [1792, 750]).astype(int)),  # 用元组储存基于摄像头的坐标   np.mutiply算的是实际坐标
              #          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
               #         )
            print(angle_kua)
            if angle2>1300:
               mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                         mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(255, 65, 105), thickness=2, circle_radius=2)
                                         )
               cv2.putText(image, "Please Laying Down",
                           (50,150),  # 用元组储存基于摄像头的坐标   np.mutiply算的是实际坐标
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),1, cv2.LINE_AA
                           )
            else:
                cv2.line(image, pshoulderleft, phipleft, (255, 0, 0), 2)
                cv2.line(image, phipleft, pkneeleft, (255, 0, 0), 2)
                cv2.line(image, pkneeleft, pankleleft, (255, 0, 0), 2)
                cv2.circle(image, pshoulderleft, 10, (0, 0, 255), cv2.FILLED)  # 10是中间的实心点
                cv2.circle(image, pshoulderleft, 15, (0, 0, 255), 2)  # 不带filled的是圈
                cv2.circle(image, phipleft, 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(image, phipleft, 15, (0, 0, 255), 2)
                cv2.circle(image, pankleleft, 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(image, pankleleft, 15, (0, 0, 255), 2)
                cv2.circle(image, pkneeleft, 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(image, pkneeleft, 15, (0, 0, 255), 2)

                cv2.putText(image, str(int(angle_kua)),
                            phipleft,  # 用元组储存基于摄像头的坐标   np.mutiply算的是实际坐标
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                cv2.putText(image, str(int(angle_xigai)),
                            pkneeleft,  # 用元组储存基于摄像头的坐标   np.mutiply算的是实际坐标
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

               # percentage =int(np.interp(angle_kua, (60, 90), (0, 100)))
                #print(angle2,percentage)
                # 进度条
              #  bar = np.interp(angle_kua, (60, 90), (200, 75))



                color=(235, 206, 135)
                #check 动作是否完成

                if angle_xigai<120:
                    if angle_kua>140:
                        color = (235, 206, 135)
                        if direction == 0:
                            count = count + 0.5
                            direction = 1

                    if angle_kua<70:
                        color = (0, 0, 255)
                        if direction ==1:
                            count=count+0.5
                            direction=0

                cTime = time.time()
                fps = 1 / (cTime - ptime)
                ptime = cTime


                print(count,direction)
                #fps

                cv2.putText(image,"FPS: "+str(int(fps)),
                            (50,400),  # 用元组储存基于摄像头的坐标   np.mutiply算的是实际坐标
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                            )
                cv2.putText(image, "done: "+str(int(count)),
                            (150,150),  # 用元组储存基于摄像头的坐标   np.mutiply算的是实际坐标
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255),5, cv2.LINE_AA
                            )


                #进度条img，p1 左上角  p2右下角
               # cv2.rectangle(image,(50,100),(75,200),color)
              #  cv2.rectangle(image,(50,int(bar)),(75,200),color,cv2.FILLED)


        except:
            pass

        # render dections  骨骼定位线
#        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                  mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
#                                  mp_drawing.DrawingSpec(color=(255, 65, 105), thickness=2, circle_radius=2)
#                                  )



        cv2.imshow('Mediapipe Feed', image)
        # cv2.imshow('Mediapipe Feed',frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cv2.waitKey(0)
cv2.destroyAllWindows()




