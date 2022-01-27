import streamlit as st
import cv2
import pickle
import numpy as np
from PIL import Image
from time import sleep
import face_recognition

print(cv2.__version__)

@st.cache
def load_image(img):
	im = Image.open(img)
	return im


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
data = pickle.loads(open('face_enc', "rb").read())

def detect_faces(image):
    new_img = np.array(image)
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
    return img,faces

def face_recog(image):
    new_img = np.array(image)
    rgb = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    # convert image to Greyscale for HaarCascade
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    encodings = face_recognition.face_encodings(rgb)
    names = []
    # loop over the facial embeddings incase
    # we have multiple embeddings for multiple fcaes
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        # set name =unknown if no encoding matches
        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            count = {}
            for i in matchedIdxs:
                name = data["names"][i]
                count[name] = count.get(name, 0) + 1
                name = max(count, key=count.get)
                names.append(name)
    for ((x, y, w, h), name) in zip(faces, names):
        cv2.rectangle(new_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(new_img, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 4)

    return new_img, faces, name


def main():
    """Face Detection App"""

    st.title("Face Detection App")
    st.text("Build with Streamlit, OpenCV & \u2764\uFE0F by Soumava Dey")

    #st.header('Face Detection Application')
    #st.write("By Soumava Dey")

    img = st.file_uploader("Uploads an image",type=['jpg','png','jpeg'])

    activities = ["Detection", "Recognition"]

    with st.sidebar.container():
        st.header("Selection of detection")
        choice = st.radio("Select Activity", activities)
        if img is not None:
            my_image = load_image(img)
            st.subheader('You uploaded below image: ')
            st.image(my_image, width=300)
            st.write("")
            st.write("")
        if st.button('About me'):
            st.markdown('Follow me on Github: [Soumava Dey](https://github.com/soudey123/)')

    if choice == "Detection":
        if img is not None:
            final_img, final_faces = detect_faces(my_image)
            st.image(final_img, width=400)
            st.success("The total faces on image are: {}".format(len(final_faces)))
            sleep(2)
            st.balloons()
        if img is None:
            st.warning('Please upload image')

    elif choice == "Recognition":
        if img is not None:
            final_img, final_faces, final_name = face_recog(my_image)
            st.success("Who is this?")
            st.image(final_img, width=400)
            st.success("This is: {}".format(str(final_name)))
            sleep(2)
            st.balloons()
        if img is None:
            st.warning('Please upload image')

if __name__ == '__main__':
    main()