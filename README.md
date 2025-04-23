# Eyesense

A web application using **ResNet50** to analyze eye images and predict possible ocular diseases. 
Built with **Streamlit**, this tool allows users to upload eye images and receive real-time predictions from a trained deep learning model.  

---

## 🚀 Getting Started  

These instructions will help you set up a local copy of the project for development and testing.  

---

## 📋 Prerequisites  

Before running the project, you need to install the following dependencies:  

- Python 3.8 or later  
- TensorFlow & Keras  
- OpenCV  
- Streamlit  

To install them, run:  

```bash
pip install -r requirements.txt
```

---

## 🔧 Installation  

Follow these steps to set up the development environment:  

1️⃣ **Clone the repository:**  
```bash
git clone https://github.com/your-username/eyesense.git
cd eyesense
```

2️⃣ **(Optional) Create a virtual environment:**  
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3️⃣ **Install dependencies:**  
```bash
pip install -r requirements.txt
```

4️⃣ **Run the Streamlit app:**  
```bash
streamlit run app/app.py
```

The application will open in your default browser.

---

## ⚙️ Running the Tests  

To ensure the project is working correctly, run the test scripts using:  

```bash
pytest tests/
```

---

### 🔩 End-to-End Tests  

These tests verify the entire workflow, from uploading an image to receiving a prediction.  

Example:  
```python
def test_prediction():
    img = load_test_image("data/example_images/eye.jpg")
    result = model.predict(img)
    assert result is not None
```

---

### ⌨️ Code Style Tests  

To maintain clean and consistent code formatting, use **Flake8**:  

```bash
flake8 app/ --max-line-length=120
```

---

## 📦 Deployment  

To deploy the application on a live server, follow these steps:  

1. Ensure all dependencies are installed in the production environment.  
2. Run `streamlit run app/app.py` inside a cloud server or container.  
3. Configure Streamlit sharing or deploy using services like **Heroku**, **AWS**, or **Google Cloud**.  

---

## 🛠️ Built With  

- **TensorFlow/Keras** - Machine learning framework  
- **Streamlit** - Web application framework  
- **OpenCV** - Image processing library  

---

## ✒️ Authors  

**Claudio Azzi**   -     [@caazzi](https://github.com/caazzi)  
**Erika Chang**    -     [@erika-chang](https://github.com/erika-chang)
**George Silva**   -     [@gbs1234](https://github.com/gbs1234)  
**Joao Sales**     -     [@masalesvic](https://github.com/masalesvic)  

