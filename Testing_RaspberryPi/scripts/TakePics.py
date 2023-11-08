import time, libcamera
from picamera2 import Picamera2, Preview
picam = Picamera2()
config = picam.create_preview_configuration(main={"size": (2592,2592)})
picam.configure(config)
picam.start_preview(Preview.QTGL)
picam.start()
picam.set_controls({"AfMode": 2})
input("Press enter to take top photo")
picam.capture_file("../Images/T.jpg")
input("Press enter to take bottom photo")
picam.capture_file("../Images/B.jpg")
picam.close()
