# Xray Lung Classifier

---
## Problem statement

---
Pneumonia is an inflammatory condition of the lung affecting primarily the small air sacs known as alveoli.Symptoms typically include some combination of productive or dry cough, chest pain, fever and difficulty breathing. The severity of the condition is variable. Pneumonia is usually caused by infection with viruses or bacteria and less commonly by other microorganisms, certain medications or conditions such as autoimmune diseases.Risk factors include cystic fibrosis, chronic obstructive pulmonary disease (COPD), asthma, diabetes, heart failure, a history of smoking, a poor ability to cough such as following a stroke and a weak immune system. Diagnosis is often based on symptoms and physical examination. Chest X-ray, blood tests, and culture of the sputum may help confirm the diagnosis.The disease may be classified by where it was acquired, such as community- or hospital-acquired or healthcare-associated pneumonia. Our task is to create a Streamlit app whichs predict whether the given images are penumonia or not.

---
## Dataset used

---
## Dataset
Chest X-ray COVID-19 Pneumonia Dataset (Kaggle): [View on Kaggle](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)

---

## Tech Stack Used
1. Python
2. Pytorch
3. Docker
4. Streamlit

---
## How to run

---
Step 1. Download the zip file

```
Download the zip file and extract it to a folder.

```

Step 2. Create a python environment.

```
python -m venv venv
venv\scripts\activate

```

Step 3. Install the requirements

```
pip install -r requirements.txt

```

Step 5. Run the applocation server

```
streamlit run app.py

```

## Models Used

---
 - Custom CNN architecture

## **xray** is the main package folder which contains

---

Components : Contains all components of Deep Learning(CV) Project

 - data_ingestion
 - data_transformation
 - model_training
 - model_evaluation
 - model_pusher

Custom Logger and Exceptions are used in the Project for better debugging purposes.

---

## Conclusion

---
 - The project we have created can also be in real-life by doctors to check whether the person is having Pneumonia or not. It will help doctors to take better decisions.