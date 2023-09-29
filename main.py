from flask import Flask,render_template
import cv2
    # Import numpy Library
import numpy as np
    # Import Keras image processing Library
from keras.preprocessing import image
    # Import Tensorflow Library
import tensorflow as tf
    # Import Client Library from twilio
from twilio.rest import Client
    # Loading the Saved Model
from PIL import Image
from keras.models import load_model
import geocoder
app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('home.html')
@app.route('/trigger_detection', methods=['GET'])
def trigger_detection():

    
    model = tf.keras.models.load_model('C:\\Users\\kaasi\\OneDrive\\Desktop\\123\\Missing_1 (1).h5')
    

    # Get the current location based on IP address
    location = geocoder.ip('me')

    # Print the latitude and longitude

    # Initialising the video
    video = cv2.VideoCapture (0)
    #Desired outputs
    name = ["Found Missing", "Normal"]
    while(True):
        success, frame = video.read()
        cv2.imwrite("ima.jpg",frame)
        img = image.load_img("ima.jpg", target_size = (64,64))
        X = image.img_to_array(img)
        x = np.expand_dims(X, axis= 0)
        pred = model.predict_step(x)
        p = int(pred[0][0])
        print(p)
        cv2.putText(frame, "Predicted Class "+str(name[p]), (100,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1)
        if pred[0][0]==0:
            # account_sid='AC8eb23713feced43a27cdd153a273de34'
            # #Twilio Account Auth Token
            # auth_token='576ff83a13ff822d0e5d62542f19b3d1'
            # #Initialise the client
            # client=Client(account_sid, auth_token)
            # # Creation of Message API
            # message=client.messages.create(
            # to="+919390310995", # FILL the contact to your desired one
            # from_="+12297158296", # Fill with your created Twilio number
            # body=" Found the Missing at"+
            # print("Latitude:", location.latlng[0])
            # print("Longitude:", location.latlng[1]) # Alert SMS Text
            # )
            # print(message.sid)
            # print("Found Missing")
            # print("SMS Sent")

            # Twilio Account SID and Auth Token
            account_sid = 'AC8eb23713feced43a27cdd153a273de34'
            auth_token = '576ff83a13ff822d0e5d62542f19b3d1'

            # Initialize the Twilio client
            client = Client(account_sid, auth_token)

            # Get the current location based on IP address
            location = geocoder.ip('me')

            # Format the live location coordinates
            latitude = location.latlng[0]
            longitude = location.latlng[1]
            location_coordinates = f"Latitude: {latitude}, Longitude: {longitude}"
            # Create and send the SMS
            message = client.messages.create(
                to="+919390310995",  # Recipient's phone number
                from_="+12297158296",  # Your Twilio phone number
                body="Found the Missing at " + location_coordinates  # SMS text with coordinates
            )

            print("SMS Sent:", message.sid)
            break
        else:
            print("Normal")
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF== ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

    return "SMS sent!👍"

if __name__ == '__main__':
    app.run()