# Face-Recognition-System
This project involves the development of a face recognition system using deep learning and computer vision techniques. The system allows users to upload images and get predictions on the identity of the person in the image.


# Face Recognition System

This project involves the development of a face recognition system using deep learning and computer vision techniques. The system allows users to upload images and get predictions on the identity of the person in the image.

## Overview

- **Data Collection & Preprocessing**: Images were collected and preprocessed to create a robust dataset.
- **Model Training**: A Convolutional Neural Network (CNN) was trained using transfer learning techniques to achieve high accuracy.
- **Evaluation**: The model's performance was assessed using metrics like accuracy, precision, and recall.
- **Image Upload and Prediction**: Users can upload images through a file upload mechanism, and the system will predict the identity of the person in the image.
- **Deployment**: The system was deployed and tested in a Google Colab environment using OpenCV.

## Tech Stack

- TensorFlow
- Keras
- OpenCV
- Google Colab

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/face-recognition-system.git
    cd face-recognition-system
    ```

2. Install the required packages:
    ```bash
    pip install tensorflow keras opencv-python
    ```

3. Download and prepare your dataset. Make sure the dataset is organized into `train` and `val` directories within a `data` directory.

## Usage

1. **Train the Model**: 
   - Ensure your dataset is properly structured in `data/train` and `data/val`.
   - Run the training script in Google Colab or your local environment.

    ```python
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    from tensorflow.keras.optimizers import Adam

    # Data generators
    train_gen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    val_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory('data/train', target_size=(100, 100), batch_size=32, class_mode='categorical')
    val_data = val_gen.flow_from_directory('data/val', target_size=(100, 100), batch_size=32, class_mode='categorical')

    num_classes = len(train_data.class_indices)

    # Model structure
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # Compile and train the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, validation_data=val_data, epochs=10)

    # Save the trained model
    model.save('face_recognition_model.h5')
    ```

2. **Run the Face Recognition Script**:
    - Use OpenCV to upload and process images.
    - Run the following script in Google Colab or your local environment.

    ```python
    import cv2
    import numpy as np
    from tensorflow.keras.models import load_model
    from google.colab import files

    # Load the trained face recognition model
    model = load_model('face_recognition_model.h5')

    # Preprocess input image
    def preprocess_image(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (100, 100))
        img_array = np.expand_dims(img, axis=0)
        return img_array / 255.0  # Normalize pixel values

    # Upload an image file
    uploaded = files.upload()

    for filename in uploaded.keys():
        # Preprocess the uploaded image
        img_array = preprocess_image(filename)

        # Perform prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        class_names = ['person1', 'person2', 'person3', 'person4', 'person5']
        predicted_name = class_names[predicted_class]

        print(f'The uploaded image is predicted to be: {predicted_name}')
    ```

## Project Structure

face-recognition-system/
├── train_model.ipynb # Notebook for training the model
├── recognize_faces.ipynb # Notebook for recognizing faces using OpenCV
├── face_recognition_model.h5 # Trained model file
├── data/
│ ├── train/ # Training data
│ └── val/ # Validation data
├── uploads/ # Directory for uploaded images
├── README.md # Project documentation
└── requirements.txt # Project dependencies



## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the community and resources that helped along the way.
- Inspired by various tutorials and documentation in the field of deep learning and computer vision.

