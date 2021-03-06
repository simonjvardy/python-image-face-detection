![My Logo](https://github.com/simonjvardy/simonjvardy/blob/main/assets/img/GitHub-name.png)

# Python - Detecting Faces in Images

## About ##

This coding example is part of a Udemy Python course using Python to detect faces in images.

---

## Technologies ##

### **Languages** ###

- [Python3](https://www.python.org/)
  - Used to create the main application functionality

### **Libraries / Packages / Modules** ###

- [OpenCV](https://github.com/opencv)
  - open-source library that includes several hundreds of computer vision algorithms.
- [OpenCV haar-cascade](https://github.com/opencv/opencv/tree/master/data/haarcascades)
  - OpenCV xml data containing trained classifiers for detecting objects of a particular type e.g. faces, people, vehicles etc.

### **Tools** ###

- [VS Code](https://code.visualstudio.com/)
  - Code Editor

---

## Deployment ##

The website was developed using VS Code & Git pushed to GitHub, which hosts the repository. I made the following steps to deploy the site:

### **Cloning python-image-face-detection** ###

#### **Prerequisites** ###

Ensure the following are installed locally on your computer:

- [Python 3.6 or higher](https://www.python.org/downloads/)
- [PIP3](https://pypi.org/project/pip/) Python package installer
- [Git](https://git-scm.com/) Version Control
- [PostgreSQL](https://www.postgresql.org/) database with pgAdmin management tool

#### **Cloning the GitHub repository** ####

- navigate to [simonjvardy/python-image-face-detection](https://github.com/simonjvardy/python-image-face-detection) GitHub repository.
- Click the **Code** button
- **Copy** the clone url in the dropdown menu
- Using your favourite IDE open up your preferred terminal.
- **Navigate** to your desired file location.

Copy the following code and input it into your terminal to clone Sportswear-Online:

```Python
git clone https://github.com/simonjvardy/python-image-face-detection.git
```

#### **Creation of a Python Virtual Environment** ####

*Note: The process may be different depending upon your own OS - please follow this [Python help guide](https://python.readthedocs.io/en/latest/library/venv.html) to understand how to create a virtual environment.*

#### **Run the application locally** ####

- To run the face detection application, enter the following command into the terminal window:

```Python
python3 face_detect.py
```

- the output image file contains the image with a green rectangle surrounding the detected face.

![Single face image](assets/img/readme_img1.jpg)

- To run the face detection application for multiple faces, enter the following command into the terminal window:

```Python
python3 multi_face_detect.py
```

- the output image file contains the image with a green rectangle surrounding the detected faces. In the example image below, OpenCv was unable to detect the face on the right chewing the newspaper page as it is partially obscured.

![Multi face image](assets/img/readme_img2.jpg)

---

## Acknowledgements ##

- [Udemy: The Python Mega Course - Build 10 Real World Applications](https://www.udemy.com/course/the-python-mega-course/) Credit: Ardit Sulce
- [OpenCV haarcascades](https://github.com/opencv/opencv/tree/master/data/haarcascades)
