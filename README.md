# Smart Classroom Face Detection and Attendance Monitor (6ZAYVEN)
CSC 126 Computer Vision – Midterm Project



## Setup
Requires Python 3.9+ and a webcam.
```bash
python -m venv venv
venv\Scripts\activate          
pip install -r requirements.txt
```
`requirements.txt` uses **opencv-contrib-python** (needed for `cv2.face`).
If you previously installed `opencv-python`, run `pip uninstall opencv-python`
first; having both causes conflicts. Linux only: `sudo apt install python3-tk`

## Run
```bash
python main.py                 # the app
python test_database.py        # optional: database + recogniser checks
python test_pipeline.py        # optional: detection test on sample images
```

## How to use
1. Register Student → enter Student No., Name, Course → face the camera.
2. Start Camera → registered students get a green box with their name.
   After ~8 consecutive matched frames the attendance is saved to the database
   (once per day). Unregistered faces get a red box and the panel shows
   **"Unrecognized face!"**.
3. **View Records** → pick a date to see every student as Present/Absent;
   **Export CSV** saves the sheet.


