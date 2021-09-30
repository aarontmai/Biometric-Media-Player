from fer import Video
from fer import FER
import os
import sys
import pandas as pd
import cv2
import numpy as np

#uses main camera 
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read camera feed")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.
out = cv2.VideoWriter('demo.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while(True):
  ret, frame = cap.read()

  if ret == True: 
    
    # Write the frame into the file 'output.avi'
    out.write(frame)

    # Display the resulting frame    
    cv2.imshow('frame',frame)

    # Press Q on keyboard to stop recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break  

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()

# Put in the location of the video file that has to be processed
location_videofile = "/home/aaron/demo.avi"

# Build the Face detection detector
face_detector = FER(mtcnn=True)
# Input the video for processing
input_video = Video(location_videofile)

# The Analyze() function will run analysis on every frame of the input video. 
# It will create a rectangular box around every image and show the emotion values next to that.
# Finally, the method will publish a new video that will have a box around the face of the human with live emotion values.
processing_data = input_video.analyze(face_detector, display=False)

# We will now convert the analysed information into a dataframe.
# This will help us import the data as a .CSV file to perform analysis over it later
vid_df = input_video.to_pandas(processing_data)
vid_df = input_video.get_first_face(vid_df)
vid_df = input_video.get_emotions(vid_df)

# Plotting the emotions against time in the video
pltfig = vid_df.plot(figsize=(20, 8), fontsize=16).get_figure()


angry = sum(vid_df.angry)
disgust = sum(vid_df.disgust)
fear = sum(vid_df.fear)
happy = sum(vid_df.happy)
sad = sum(vid_df.sad)
surprise = sum(vid_df.surprise)
neutral = sum(vid_df.neutral)

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotions_values = [angry, disgust, fear, happy, sad, surprise, neutral]

score_comparisons = pd.DataFrame(emotions, columns = ['Human Emotions'])
score_comparisons['Emotion Value from the Video'] = emotions_values
score_comparisons
#creating dictionary to find dominant mood
iterator = zip(emotions, emotions_values)
mood = dict(iterator)
z = max(emotions_values)
dominantMood = (list(mood.keys())[list(mood.values()).index(z)])

#convert emotional data to csv 
score_comparisons.to_csv(r'/home/aaron/export_dataframe.csv', index = False, header = True)
print()
print(">>> Your main mood is " + str(dominantMood) + ", I'll play you some fitting music!!\n")
print("Here's what we think what you're feeling\n")
print(score_comparisons)

#hard coded mp3 playlist player for demo
if dominantMood == 'Happy':
    os.chdir("Happy")
    os.system("mpg123 -Z *.mp3")
elif dominantMood == 'Sad':
    os.chdir("Sad")
    os.system("mpg123 -Z *.mp3")
elif dominantMood == 'Angry':
    os.chdir("Angry")
    os.system("mpg123 *.mp3")
elif dominantMood == 'Fear':
    os.chdir("Fear")
    os.system("mpg123 *.mp3")
elif dominantMood == 'Suprise':
    os.chdir("Suprise")
    os.system("mpg123 *.mp3")
elif dominantMood == 'Disgust':
    os.chdir("Disgust")
    os.system("mpg123 *.mp3")
elif dominantMood == 'Neutral':
    os.chdir("Neutral")
    os.system("mpg123 *.mp3")

